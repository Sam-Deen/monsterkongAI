import numpy as np
import torch
from dqn_agent import DQNAgent
from ple import PLE
from ple.games.monsterkong import MonsterKong
import sys

rewards = {
    "positive": 5,
    "win": 0, # last step seems to be "a" when saving princess, giving a score will massively favour "a" and brick the AI
    "negative": -25,
    "tick": -0.1
}
totalEpisodes = 0

def get_game_state_vector(game_state):
    return np.array([
        game_state["player_x"],
        game_state["player_y"],
        float(game_state["on_ladder"]),
        float(game_state["on_ground"]),
        game_state["closest_ladder_x"],
        game_state["closest_coin_x"],
        game_state["closest_coin_y"],
    ], dtype=np.float32)

def train_agent(env, agent, episodes):
    for e in range(episodes):
        game_state = env.getGameState()
        state = get_game_state_vector(game_state)

        total_reward = 0
        best_y = game_state["player_y"]
        timer = 0

        while not env.game_over() and timer < 900:
            action = agent.select_action(state)
            reward = env.act(action)

            new_state = env.getGameState()
            next_state = get_game_state_vector(new_state)

            if new_state['on_ladder']:
                reward += 1

            climb_reward = 0
            if new_state["player_y"] < best_y and new_state["on_ladder"]:
                climb_reward = (best_y - new_state["player_y"]) * 3
                best_y = new_state["player_y"]
            reward += climb_reward
            done = env.game_over()

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            timer += 1
            agent.replay()
        if e % 5 == 0:
            agent.update_target_model()

        agent.decay_epsilon()
        env.reset_game()
        print(f"Episode {e + 1}/{episodes} - "
              f"Total Reward: {total_reward:.2f} | "
              f"Epsilon: {agent.epsilon:.4f} | "
              f"Memory: {len(agent.memory)}/{agent.memory.maxlen} | ")
        global totalEpisodes
        totalEpisodes += 1
        print(f"Total episodes: {totalEpisodes}")
if __name__ == "__main__":
    game = MonsterKong()
    env = PLE(game, fps=30, display_screen=False, reward_values=rewards)
    env.init()

    action_set = env.getActionSet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available, check torch installation")

    agent = DQNAgent(input_dim=7, action_set=action_set, device=device)

    try:
        totalEpisodes = agent.load("checkpoint.pkl")
    except FileNotFoundError:
        print("No previous checkpoint found. Starting fresh.")
    except RuntimeError:
        print("Error decoding model. Starting fresh.")
    try:
        train_agent(env, agent, episodes=15000)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Saving...")
        agent.save("checkpoint.pkl", totalEpisodes)
        sys.exit(0)
