import numpy as np
import torch
import cv2
from dqn_agent import DQNAgent
from ple import PLE
from ple.games.monsterkong import MonsterKong
import sys

rewards = {
    "positive": 0,
    "win": 100,
    "negative": -25,
    "tick": -0.01
}
totalEpisodes = 0

def preprocess_frame(frame, size):
    frame = cv2.resize(frame, size)
    frame = np.array(frame, dtype=np.float32) / 255.0
    return frame[np.newaxis, :, :]


def train_agent(env, agent, episodes):
    for e in range(episodes):
        raw_frame = env.getScreenGrayscale()
        preprocessed = preprocess_frame(raw_frame, agent.state_size)

        agent.state_stack.clear()
        for _ in range(agent.frame_stack):
            agent.state_stack.append(preprocessed)

        state = np.concatenate(agent.state_stack, axis=0)  # shape: (4, H, W)

        total_reward = 0
        best_y = env.getGameState()["player_y"]
        timer = 0

        while not env.game_over() and timer < 900:
            action = agent.select_action(state)
            reward = env.act(action)

            new_y = env.getGameState()["player_y"]

            if env.getGameState()['onladder']:
                reward += 0.1

            climb_reward = 0
            if new_y < best_y:
                climb_reward = (best_y - new_y) * 0.5
                best_y = new_y
            reward += climb_reward

            raw_frame = env.getScreenGrayscale()
            next_frame = preprocess_frame(raw_frame, agent.state_size)
            agent.state_stack.append(next_frame)
            next_state = np.concatenate(agent.state_stack, axis=0)

            done = env.game_over()

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            timer += 1
            agent.replay()
        if e % 10 == 0:
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
    env = PLE(game, fps=30, display_screen=True, reward_values=rewards)
    env.init()

    action_set = env.getActionSet()
    state_size = (99, 96)  # Any multiple of 33x32 block sizes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available, check torch installation")
    agent = DQNAgent(state_size, action_set, device)
    try:
        totalEpisodes = agent.load("checkpoint")
    except FileNotFoundError:
        print("No previous checkpoint found. Starting fresh.")
    except RuntimeError:
        print("Error decoding model. Starting fresh.")
    try:
        train_agent(env, agent, episodes=15000)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Saving...")
        agent.save("checkpoint", totalEpisodes)
        sys.exit(0)
