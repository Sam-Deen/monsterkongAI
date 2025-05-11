import numpy as np
import torch
from dqn_agent import DQNAgent
from ple import PLE
from ple.games.monsterkong import MonsterKong
import sys

# Define custom rewards for the game
rewards = {
    "positive": 2.5,    # coin
    "win": 500,         # save princess
    "negative": -25,    # fireball
    "tick": -0.1        # every frame
}

total_episodes = 0
new_floor_reward = 50  # Reward for reaching a new floor


def get_state_vector(game_data):
    """
    Convert the game data dictionary to a vector for the DQN agent.
    """
    return np.array([
        game_data["player_x"],
        game_data["player_y"],
        float(game_data["on_ladder"]),
        float(game_data["on_ground"]),
        game_data["closest_ladder_x"],
        game_data["closest_coin_x"],
        game_data["closest_coin_y"],
    ], dtype=np.float32)

def train_agent(env, agent, episodes):
    """
    Train the DQN agent over a specified number of episodes.
    """
    global total_episodes

    for episode in range(episodes):
        # Initialize game data and variables for the current episode
        initial_game_data = env.getGameState()
        current_state = get_state_vector(initial_game_data)
        total_reward = 0
        best_y = initial_game_data["player_y"]  # Track player's highest y-coordinate
        timer = 0  # Prevent infinite loops

        while not env.game_over() and timer < 3000:
            # Agent selects an action
            action = agent.select_action(current_state)
            reward = env.act(action)

            # Process the updated game data
            updated_game_data = env.getGameState()
            updated_state = get_state_vector(updated_game_data)

            # Bonus reward for climbing higher on a ladder
            climb_reward = 0
            if updated_game_data["player_y"] < best_y and updated_game_data["on_ladder"]:
                climb_reward = (best_y - updated_game_data["player_y"]) * 0.5
                best_y = updated_game_data["player_y"]

            reward += climb_reward

            # bonus reward for getting to new floor
            floors = [360, 285, 210, 135, 60]  # Y-coordinates for floors
            reached_floors = set()  # Track milestones reached in this episode
            for floor in floors:
                if updated_game_data["player_y"] < floor and floor not in reached_floors:
                    reward += new_floor_reward
                    reached_floors.add(floor)


            # Ladder lingering penalty
            ladder_penalty = -0.05 if updated_game_data["on_ladder"] else 0

            reward += ladder_penalty

            done = env.game_over()

            # Store the experience in the agent's memory
            agent.remember(current_state, action, reward, updated_state, done)
            current_state = updated_state

            total_reward += reward
            timer += 1

            # Perform a training step
            agent.replay()

            # Break early if the win reward is achieved
            if reward >= rewards["win"]:
                break


        # Update the target model
        if episode % 10 == 0:
            agent.update_target_model()

        # Decay exploration rate
        agent.decay_epsilon()

        env.reset_game()

        # Log progress
        print(f"Episode {episode + 1}/{episodes} - "
              f"Total Reward: {total_reward:.2f} | "
              f"Epsilon: {agent.epsilon:.4f} | "
              f"Memory: {len(agent.memory)}/{agent.memory.maxlen}")

        total_episodes += 1
        print(f"Total episodes so far: {total_episodes}")

if __name__ == "__main__":
    # Initialize the MonsterKong game environment
    game = MonsterKong()
    env = PLE(game, fps=30, display_screen=True, reward_values=rewards)
    env.init()

    # Get the set of possible actions
    action_set = env.getActionSet()

    # Determine the computing device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available, check torch installation")

    # Initialize the DQN agent
    agent = DQNAgent(input_dim=7, action_set=action_set, device=device)

    # Attempt to load the agent's prior training checkpoint
    try:
        total_episodes = agent.load("checkpoint.pkl")
    except FileNotFoundError:
        print("No previous checkpoint found. Starting fresh.")
    except RuntimeError:
        print("Error decoding model. Starting fresh.")

    # Train the agent, handling interruptions gracefully
    try:
        train_agent(env, agent, episodes=15000)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Saving...")
        agent.save("checkpoint.pkl", total_episodes)
        sys.exit(0)
