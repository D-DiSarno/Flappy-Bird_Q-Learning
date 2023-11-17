from FlappyBird import flappy as fp
from FlappyBird.flappy import PlayGame, TrainGame, EvaluateGame
import os

file_dir = os.path.dirname(os.path.realpath(__file__))

gm = fp.gameManager(PlayGame(file_dir=file_dir))
gm.play()
