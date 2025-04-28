from ple.games.monsterkong import MonsterKong
from ple import PLE
from qlearning_agent import QLearningAgent


def run_monsterkong():
    game = MonsterKong()
    env = PLE(game, display_screen=True)
    env.init()
    actions = env.getActionSet()
    print(f"Available actions: {actions}")

    # Initialize agent (but without learning)
    agent = QLearningAgent(
        action_space=actions,
        downsample_size=(64, 64)  # Downsample to 64x64 for efficiency
    )

    # Run for a set number of steps or until game over
    for _ in range(5000):
        if env.game_over():  # Check if the game is over
            env.reset_game()  # Reset the game if it's over

        frame = env.getScreenRGB()  # Get the current screen as an RGB image

        # Get action from the agent (without learning)
        action = agent.act(frame)
        print(f"Action chosen: {action}")

        # Perform the action in the environment
        reward = env.act(action)
        print(f"Reward: {reward}")

        # Optionally, visualize or log the state, action, and reward
        # No learning happening here, agent is simply acting


def main():
    run_monsterkong()


if __name__ == '__main__':
    main()
