import math
import pickle

import numpy as np
import torch
from dqn_agent import DQNAgent
from ple import PLE
from ple.games.monsterkong import MonsterKong
import sys

# Define custom rewards for the game
rewards = {
    "positive": 0,    # coin
    "win": 500,         # save princess
    "negative": -1,    # fireball
    "tick": -0.1        # every frame
}

total_episodes = 0
new_floor_reward = 50  # Reward for reaching a new floor
ladder_reward = 0.1

reward_history = []
epsilon_history = []


def get_state_vector(state):
    """
    Convert the game data dictionary to a vector for the DQN agent.
    """
    return np.array([
        state["player_x"],
        state["player_y"],
        float(state["on_ladder"]),
        float(state["on_ground"]),
        state["closest_ladder_x"],
        state["closest_fireball_x"],
        state["closest_fireball_y"],
    ], dtype=np.float32)

def train_agent(env, agent, episodes):
    """
    Train the DQN agent over a specified number of episodes.
    """
    global total_episodes

    for e in range(episodes):
        initial_game_state = env.getGameState()
        initial_vector_state = get_state_vector(initial_game_state)

        total_reward = 0
        reached_floors = set()  # Track milestones reached in this episode

        best_y = initial_game_state["player_y"]
        timer = 0 # Prevent infinite loops while no lose state

        while not env.game_over() and timer < 1000:
            # Agent selects an action
            action = agent.select_action(initial_vector_state)
            reward = env.act(action)

            # Process the updated game data
            post_action_game_state = env.getGameState()
            post_action_vector_state = get_state_vector(post_action_game_state)

            # Bonus reward for climbing higher on a ladder
            climb_reward = 0
            if post_action_game_state["player_y"] < best_y and post_action_game_state["on_ladder"]:
                climb_reward = (best_y - post_action_game_state["player_y"]) * 0.5
                best_y = post_action_game_state["player_y"]
            reward += climb_reward

            # reward += (435 - post_action_game_state["player_y"]) * 0.01

            # bonus reward for getting to new floor
            floors = [360, 285, 210, 135, 60]  # Y-coordinates for floors

            for floor in floors:
                if post_action_game_state["player_y"] < floor and floor not in reached_floors:
                    reward += new_floor_reward
                    reached_floors.add(floor)

            # bonus reward for just being on a ladder
            if post_action_game_state["on_ladder"]:
                reward += ladder_reward

            done = env.game_over()

            # print(f"initial vectors: {initial_vector_state} post action: {post_action_vector_state} action taken: {chr(action) if action is not None else "none"} reward: {round(reward, 3)} ")

            def contains_nan(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    return np.isnan(x).any()
                if isinstance(x, (float, int)):
                    return math.isnan(x)
                return False

            if (
                    contains_nan(initial_vector_state)
                    or contains_nan(post_action_vector_state)
                    or contains_nan(reward)
            ):
                print("⚠️ NaN detected!")
                print(
                    f"initial vectors: {initial_vector_state} post action: {post_action_vector_state} action taken: {chr(action) if action is not None else 'none'} reward: {round(reward, 3) if not contains_nan(reward) else 'NaN'}")

            agent.remember(initial_vector_state, action, reward, post_action_vector_state, done)
            initial_vector_state = post_action_vector_state
            total_reward += reward
            timer += 1

            # Perform a training step
            agent.replay()

            # Break early if the win reward is achieved
            if reward >= rewards["win"]:
                break

        # Update the target model
        if e % 1 == 0:
            agent.update_target_model()
        agent.decay_epsilon()
        env.reset_game()

        reward_history.append(total_reward)
        epsilon_history.append(agent.epsilon)

        print(f"Episode {e + 1}/{episodes} - "
              f"Total Reward: {total_reward:.2f} | "
              f"Epsilon: {agent.epsilon:.4f} | "
              f"Memory: {len(agent.memory)}/{agent.memory.maxlen} | ")
        total_episodes += 1
        print(f"Total episodes: {total_episodes}")

def save_training_logs(reward_history, epsilon_history, filename="training_log.pkl"):
    log_data = {
        "rewards": reward_history,
        "epsilons": epsilon_history
    }
    with open(filename, "wb") as f:
        pickle.dump(log_data, f)
    print(f"[INFO] Training logs saved to {filename}")

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
    agent = DQNAgent(input_dim=len(env.getGameState()), action_set=action_set, device=device)

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
        save_training_logs(reward_history, epsilon_history)
        sys.exit(0)