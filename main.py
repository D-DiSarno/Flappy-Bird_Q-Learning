from FlappyBird import flappy as fp
from FlappyBird.flappy import PlayGame, TrainGame, EvaluateGame
import os
import Deep_Q_Learning as qm
import Q_Learning as qb

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    print('0 - Train')
    print('1 - Evaluate')
    print('2 - Play')
    typeGame = int(input('Type of game: '))
    while True:
        if typeGame == 0:
            gm = fp.gameManager(TrainGame(file_dir=file_dir))
            agent = qb.QLearningAgent(gm)
            epochs = int(input('For how many epochs? '))
            qb.train(agent, epochs=epochs, plotting_scores=True)
            break
        elif typeGame == 1:
            fps = int(input('Insert framerate: '))
            gm = fp.gameManager(EvaluateGame(file_dir=file_dir, fps=fps))
            agent = qb.QLearningAgent(gm)
            filename = input('Relative path to model folder: ')
            agent.load_model(filename=filename+"/")
            epochs = int(input('For how many epochs? '))
            qb.evaluate(agent, epochs=epochs)
            break
        elif typeGame == 2:
            gm = fp.gameManager(PlayGame(file_dir=file_dir))
            gm.play()
            break
        else:
            typeGame = int(input('Please, insert a supported type of game: '))
