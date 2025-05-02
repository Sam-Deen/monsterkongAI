from ple.games.monsterkong import MonsterKong
from ple import PLE
from qlearning_agent import QLearningAgent
from dqn_agent import DQNAgent
rewards = {
    "positive": 5,
    "win": 100,
    "negative": -25,
    "tick": -0.01
}

def train_monsterkong():
    game = MonsterKong()
    env = PLE(game, display_screen=True, reward_values=rewards)
    env.init()
    actions = env.getActionSet()

    print(f"Available actions: {actions}")
    # Initialize agent
    agent = DQNAgent(#QLearningAgent(
        action_space=actions,
        downsample_size=(32, 32)  # Downsample to 42x42 for efficiency (will change when home with desktop)
    )
    max_frames = 1000
    episode = 0
    # Run for a set number of steps or until game over
    while True:
        if env.game_over():  # Check if the game is over
            env.reset_game()  # Reset the game if it's over

        frame = env.getScreenGrayscale()  # Get the current screen as a greyscale image
        state = agent.preprocess(frame)  # Preprocess the frame into a workable state

        done = False
        total_reward = 0
        frame_count = 0

        while not done:
            # Step 1: Choose action based on current state
            action = agent.act(state)  # Select action (e.g., 'w', 'a', 's', 'd', 'space', 'nothing')

            # Step 2: Perform the action in the environment
            reward = env.act(action)  # Perform the action and get reward

            # Step 3: Get the next state and preprocess it
            next_frame = env.getScreenRGB()  # Get the next screen
            next_state = agent.preprocess(next_frame)  # Preprocess the next state

            # Step 4: Check if the game is over
            done = env.game_over()

            # Step 5: Learn from the experience (state, action, reward, next_state, done)
            agent.learn(state, action, reward, next_state, done)

            # Update the state
            state = next_state
            total_reward += reward
            frame_count += 1
            print(f"State: {env.getGameState()}")

            if frame_count >= max_frames:
                done = True
                env.reset_game()
        print(f"Episode {episode + 1} finished. Total reward: {total_reward}")
        max_frames += 1
        episode += 1

def main():
    train_monsterkong()


if __name__ == '__main__':
    main()