from ple.games.monsterkong import MonsterKong
from ple import PLE

game = MonsterKong() # create new game instance
env = PLE(game, display_screen=True) # create new environment that the game will live in with display on
env.init() # start the game

for _ in range(1000): # for the first 1000 frames
    if env.game_over(): # check if the player died
        env.reset_game() # reset game
    action = env.getActionSet()[0]  # choose  first action in array of all possible actions, in this case a
    env.act(action) # do that action in the game

