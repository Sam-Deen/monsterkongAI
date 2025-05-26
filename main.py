import matplotlib.pyplot as plt
import csv
import os
import math
import numpy as np
import torch
from dqn_agent import DQNAgent
from ple import PLE
from ple.games.monsterkong import MonsterKong
import sys
from datetime import datetime
import cv2
from io import BytesIO
from PIL import Image

reward_history = []
epsilon_history = []


# Define custom rewards for the game
rewards = {
    "positive": 0,    # coin
    "win": 500,         # save princess
    "negative": -1,    # fireball
    "tick": -0.1        # every frame
}

total_episodes = 0

ladder_reward = 0.1
climb_multiplier = 10.0
correct_direction = 0.5

log_file = "training_log.csv"


def decay_shaping_reward(base_value, decay_rate, episode, min_value=0.0):
    return max(min_value, base_value * (1 - decay_rate * episode))

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


def plot_network_weights(agent, input_labels, action_chars):
    # Get first layer weights
    weights = agent.model.feature[0].weight.data.cpu().numpy()

    input_count = len(input_labels)
    output_count = len(action_chars)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Positions
    input_positions = [(0.1, 1 - (i + 1) / (input_count + 1)) for i in range(input_count)]
    output_positions = [(0.9, 1 - (i + 1) / (output_count + 1)) for i in range(output_count)]

    # Draw connections
    for i, (x1, y1) in enumerate(input_positions):
        for j, (x2, y2) in enumerate(output_positions):
            weight = weights[j][i]
            color = 'red' if weight > 0 else 'blue'
            linewidth = min(5, max(0.5, abs(weight) * 2))
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.7)

    # Input labels
    for i, (x, y) in enumerate(input_positions):
        ax.text(x - 0.05, y, input_labels[i], ha='right', va='center')

    # Output labels
    for j, (x, y) in enumerate(output_positions):
        action_char = chr(action_chars[j]) if isinstance(action_chars[j], int) else str(action_chars[j])
        ax.text(x + 0.05, y, action_char, ha='left', va='center')

    plt.title("DQN First Layer Connections")

    # Convert the Matplotlib figure to a NumPy array image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_pil = Image.open(buf)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Show with OpenCV
    cv2.imshow("Network Weights", img_cv)
    cv2.waitKey(1)

def show_q_values_window(q_values, action_chars=None, chosen_action=None):
    num_actions = len(q_values)
    width = 400
    height = 50 + 40 * num_actions
    img = np.zeros((height, width, 3), dtype=np.uint8)  # black background

    max_q = max(q_values)
    max_bar_length = 300
    q_scale_max = max(max_q, 1000)
    scale = max_bar_length / q_scale_max if q_scale_max > 0 else 1.0

    for i, q in enumerate(q_values):
        bar_length = int(q * scale)
        y = 40 + i * 40

        # Draw background bar (grey)
        cv2.rectangle(img, (80, y - 15), (380, y + 15), (50, 50, 50), -1)

        # Determine the char label for the action
        if action_chars:
            char = chr(action_chars[i]) if isinstance(action_chars[i], int) else action_chars[i]
        else:
            char = None

        # Bar color: green if this action's char matches chosen_action, else blue
        if chosen_action is not None and char is not None and ord(char) == chosen_action:
            bar_color = (0, 255, 0)  # green
        else:
            bar_color = (255, 0, 0)  # blue

        # Draw bar
        cv2.rectangle(img, (80, y - 15), (80 + bar_length, y + 15), bar_color, -1)

        # Label text
        label = char if char else f"Action {i}"
        text = f"{label}: {q:.3f}"

        cv2.putText(img, text, (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Q-Values", img)
    cv2.waitKey(1)

def initialize_logging():
    # Setup logging
    write_header = not os.path.exists(log_file)
    if write_header:
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "TotalReward", "Epsilon"])


def apply_rewards(initial_game_state=None, post_action_game_state=None, best_y=None):

    reward = 0
    # Bonus reward for climbing higher on a ladder
    if post_action_game_state["player_y"] < best_y and post_action_game_state["on_ladder"]:
        climb_reward = (best_y - post_action_game_state["player_y"]) * climb_multiplier or 0
        best_y = post_action_game_state["player_y"]
        reward += climb_reward

    # bonus reward for just being on a ladder
    if post_action_game_state["on_ladder"]:
        reward += ladder_reward

    # bonus reward for going towards the ladder
    prev_distance = abs(initial_game_state["player_x"] - initial_game_state["closest_ladder_x"])
    new_distance = abs(post_action_game_state["player_x"] - post_action_game_state["closest_ladder_x"])

    if new_distance < prev_distance:
        reward += correct_direction

    return reward, best_y


def run_single_episode(env, agent):
    initial_game_state = env.getGameState()
    initial_vector_state = get_state_vector(initial_game_state)

    total_reward = 0
    best_y = initial_game_state["player_y"]
    timer = 0
    episode_memory = []

    # Decay intermediate shaping rewards
    # if wonned:
    #    current_ladder_reward = decay_shaping_reward(ladder_reward, 0.005, total_episodes)
    #    current_direction_reward = decay_shaping_reward(correct_direction, 0.005, total_episodes)

    while not env.game_over() and timer < 2501:
        action = agent.select_action(initial_vector_state)
        reward = env.act(action)

        q_values = get_q_values(agent, initial_vector_state)

        if env.display_screen:
            show_q_values_window(q_values, action_chars=env.getActionSet(), chosen_action=action)

        post_action_game_state = env.getGameState()
        post_action_vector_state = get_state_vector(post_action_game_state)

        shaped_reward, best_y = apply_rewards(initial_game_state=initial_game_state, post_action_game_state=post_action_game_state, best_y=best_y)
        reward += shaped_reward
        done = env.game_over()

        # print(f"initial vectors: {initial_vector_state} post action: {post_action_vector_state} action taken: {chr(action) if action is not None else "none"} reward: {round(reward, 3)} ")
        if contains_nan(initial_vector_state) or contains_nan(post_action_vector_state) or contains_nan(reward):
            print("⚠️ NaN detected!")
            print(f"initial vectors: {initial_vector_state} post action: {post_action_vector_state} action taken: {chr(action) if action is not None else 'none'} reward: {round(reward, 3) if not contains_nan(reward) else 'NaN'}")

        agent.remember_single(initial_vector_state, action, reward, post_action_vector_state, done, to_win_memory=False)
        episode_memory.append((initial_vector_state, action, reward, post_action_vector_state, done))

        initial_vector_state = post_action_vector_state
        initial_game_state = post_action_game_state
        total_reward += reward
        timer += 1

        if timer % 500 == 0:
            agent.update_target_model()
            if env.display_screen:
                plot_network_weights(agent, list(initial_game_state.keys()), env.getActionSet())

        agent.replay()

        if reward >= rewards["win"]:
            if hasattr(agent, "win_memory"):
                agent.remember_batch(episode_memory, to_win_memory=True)
            break

    return total_reward

def get_q_values(agent, state_vector):
    state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        q_values_tensor = agent.model(state_tensor)
    return q_values_tensor.squeeze(0).cpu().numpy()

def train_agent(env, agent, episodes):
    global total_episodes

    # wonned = False
    # current_ladder_reward = ladder_reward
    # current_direction_reward = correct_direction

    initialize_logging()

    for e in range(episodes):
        total_reward = run_single_episode(env=env, agent=agent)
        total_episodes += 1


        # Update the target model
        # if e % 5 == 0:
        #     agent.update_target_model()
        if total_episodes % 500 == 0:
            agent.epsilon  = max(agent.epsilon, 0.3)

        agent.decay_epsilon()
        env.reset_game()

        print(f"Episode {e + 1}/{episodes} - "
              f"Total Reward: {total_reward:.2f} | "
              f"Epsilon: {agent.epsilon:.4f} | "
              f"Memory: {len(agent.memory)}/{agent.memory.maxlen} | "
              f"Wom Memory: {len(agent.win_memory)}/{agent.win_memory.maxlen}")
        total_episodes += 1
        print(f"Total episodes: {total_episodes}")

        # Log to CSV
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([total_episodes, total_reward, agent.epsilon])

def plot_csv(agent):
    # --- Plotting from CSV ---
    plot_episodes = []
    plot_rewards = []
    plot_epsilons = []

    with open(log_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            plot_episodes.append(int(row["Episode"]))
            plot_rewards.append(float(row["TotalReward"]))
            plot_epsilons.append(float(row["Epsilon"]))

    plt.figure(figsize=(12, 5))

    # Total reward
    plt.subplot(1, 2, 1)
    plt.plot(plot_episodes, plot_rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards Over Time')
    plt.legend()

    # Epsilon decay
    plt.subplot(1, 2, 2)
    plt.plot(plot_episodes, plot_epsilons, label='Epsilon', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Over Time')
    plt.legend()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"training_plot_{timestamp}.png"

    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.show()

    if len(agent.loss_history) > 0:
        plt.figure()
        plt.plot(agent.loss_history, label='Training Loss', alpha=0.6)
        plt.xlabel('Training Step')
        plt.ylabel('MSE Loss')
        plt.title('Loss During Training')
        plt.legend()
        loss_filename = f"loss_plot_{timestamp}.png"
        plt.savefig(loss_filename)
        plt.show()

def contains_nan(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.isnan(x).any()
    if isinstance(x, (float, int)):
        return math.isnan(x)
    return False

if __name__ == "__main__":
    # Initialize the MonsterKong game environment
    game = MonsterKong()
    game_env = PLE(game, fps=30, display_screen=False, reward_values=rewards)
    game_env.init()

    # Get the set of possible actions
    action_set = game_env.getActionSet()

    # Determine the computing device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is not available, check torch installation")

    # Initialize the DQN agent
    dqnagent = DQNAgent(input_dim=len(game_env.getGameState()), action_set=action_set, device=device)

    # Attempt to load the dqnagent's prior training checkpoint
    try:
        total_episodes = dqnagent.load("checkpoint.pkl")
    except FileNotFoundError:
        print("No previous checkpoint found. Starting fresh.")
    except RuntimeError:
        print("Error decoding model. Starting fresh.")

    # Train the dqnagent, handling interruptions gracefully
    try:
        train_agent(game_env, dqnagent, episodes=15000)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Saving...")
        dqnagent.save("checkpoint.pkl", total_episodes)
        plot_csv(dqnagent)
        sys.exit(0)