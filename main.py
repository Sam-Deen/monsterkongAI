import numpy as np
import torch
import cv2
from dqn_agent import DQNAgent
from ple import PLE
from ple.games.monsterkong import MonsterKong

rewards = {
    "positive": 0,
    "win": 100,
    "negative": -25,
    "tick": -0.01
}


def preprocess_frame(frame, size):
    frame = cv2.resize(frame, size)
    frame = np.array(frame, dtype=np.float32) / 255.0
    return frame[np.newaxis, :, :]


def train_agent(env, agent, episodes):
    for e in range(episodes):
        state = preprocess_frame(env.getScreenGrayscale(), agent.state_size)
        total_reward = 0
        best_y = env.getGameState()["player_y"]

        while not env.game_over():
            action = agent.select_action(state)
            reward = env.act(action)

            new_y = env.getGameState()["player_y"]
            climb_reward = 0
            if new_y < best_y:
                climb_reward = (best_y - new_y) * 0.5
                best_y = new_y
            reward += climb_reward

            next_state = preprocess_frame(env.getScreenGrayscale(), agent.state_size)
            done = env.game_over()

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay()

        agent.update_target_model()
        env.reset_game()
        print(f"Episode {e + 1}/{episodes} - Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    game = MonsterKong()
    env = PLE(game, fps=30, display_screen=True, reward_values=rewards)
    env.init()

    action_set = env.getActionSet()
    state_size = (99, 96)  # Any multiple of 33x32 block sizes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available, check torchvision installation")
    agent = DQNAgent(state_size, action_set, device)
    train_agent(env, agent, episodes=1000)
