from FlappyBird import flappy as fp
from FlappyBird.flappy import PlayGame, TrainGame, EvaluateGame
import os
import Q_learning_Manager as qm
import Q_Learning_Base as qb


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    gm = fp.gameManager(TrainGame(file_dir=file_dir))
    agent = qm.QLearningAgent(gm)
    qm.train(agent, epochs=2500, plotting_scores=True)
