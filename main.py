import os

import torch
from IPython import display
from matplotlib import pyplot as plt
from tqdm import tqdm

from models.deep_q_learning import DeepQLearning
from models.q_learning import QLearning
from models.sarsa import SARSA
from agent import FlappyAgent
from ple import PLE
from ple.games.flappybird import FlappyBird


def plot(scores, mean_scores, type_game):
    display.clear_output(wait=True)
    plt.clf()
    plt.title(type_game)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def train(nb_episodes, agent, model_name):
    reward_values = agent.reward_values()

    env = PLE(
        FlappyBird(),
        fps=30,  # di base 30
        display_screen=False,
        force_fps=True,
        rng=None,
        reward_values=reward_values,
    )
    env.init()

    frames = 0
    total_rewards = []
    scores = []
    game_scores = []
    total = nb_episodes

    plot_scores = []
    plot_mean_scores = []

    score = 0
    game_score = 0
    with tqdm(total=nb_episodes, position=0, leave=True) as pbar:
        while nb_episodes > 0:
            # pick an action
            state = env.game.getGameState()
            action = agent.training_policy(state)

            # step the environment
            reward = env.act(env.getActionSet()[action])

            # let the agent observe the current state transition
            new_state = env.game.getGameState()
            agent.observe(state, action, reward, new_state, env.game_over())
            frames += 1

            score += reward
            game_score += 1 if int(reward) > 0 else 0

            # reset the environment if the game is over
            if env.game_over():
                if nb_episodes % 1000 == 0 and scores != []:
                    print(
                        f"\nREWARDS: [MAX: {max(scores)}] [AVG: {sum(scores) / len(scores)}]"
                    )
                    print(
                        f"SCORES: [MAX: {max(game_scores)}] [AVG: {sum(game_scores) / len(game_scores)}]\n"
                    )
                    plot_scores.append(game_score)
                    mean_score = sum(game_scores) / len(game_scores)
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores, "Training...")
                    scores = []
                    game_scores = []
                if score > 1000 and score > max_score:
                    agent.save_model()
                    max_score = score

                scores.append(score)
                game_scores.append(game_score)
                total_rewards.append(score)
                env.reset_game()
                pbar.update(1)
                nb_episodes -= 1
                score = 0
                game_score = 0

        print(f"\nMax Reward: {max(scores)}")
        print(f"Avg Reward for {total} episodes: {sum(scores) / len(scores)}")
        print(f"Max Score: {max(game_scores)}")
        print(f"Avg Score for {total} episodes: {sum(game_scores) / len(game_scores)}")

    model_dir = f"pretrained_models/model_{model_name}_{str(total)}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    agent.save_model(filename=os.path.join(model_dir, f"q_table_{str(total)}"))

    plt.plot(game_scores, label="Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(model_dir, f"plot_{str(total)}.png"))
    plt.show()


def test(nb_episodes, agent):
    reward_values = agent.reward_values()

    env = PLE(
        FlappyBird(),
        fps=30,
        display_screen=False,
        force_fps=True,
        rng=None,
        reward_values=reward_values,
    )
    env.init()
    scores = []  # Punteggi dovuti alla reward
    game_scores = []  # Punteggi ottenuti nel gioco

    max_score = float("-inf")
    score = 0
    game_score = 0
    with tqdm(total=nb_episodes, position=0, leave=True) as pbar:
        while nb_episodes > 0:
            action = agent.policy(env.game.getGameState())
            reward = env.act(env.getActionSet()[action])
            score += reward
            game_score += 1 if int(reward) > 0 else 0

            # reset the environment if the game is over
            if env.game_over():
                if score > 1000 and score > max_score:
                    max_score = score

                if nb_episodes % 20 == 0 and scores != []:
                    print(
                        f"\nREWARDS: [MAX: {max(scores)}] [AVG: {sum(scores) / len(scores)}]"
                    )
                    print(
                        f"SCORES: [MAX: {max(game_scores)}] [AVG: {sum(game_scores) / len(game_scores)}]\n"
                    )

                pbar.update(1)
                scores.append(score)
                game_scores.append(game_score)
                env.reset_game()
                nb_episodes -= 1
                score = 0
                game_score = 0

    print(f"\nMax Reward: {max(scores)}")
    print(f"Max Score: {max(game_scores)}")

    plt.plot(game_scores, label="Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('0 - Train')
    print('1 - Test')
    typeGame = int(input('Select an action: '))
    while True:
        if typeGame == 0:
            print('0 - Q-Learning')
            print('1 - DeepQ-Learning')
            print('2 - SARSA')
            typeLearning = int(input('Select a type of learning: '))
            while True:
                if typeLearning == 0:
                    agent = FlappyAgent(learner=QLearning())
                    break
                elif typeLearning == 1:
                    agent = FlappyAgent(learner=DeepQLearning(8, 2))
                    break
                elif typeLearning == 2:
                    agent = FlappyAgent(learner=SARSA())
                    break
                else:
                    typeLearning = int(input('Please, insert a supported action: '))
            episodes = int(input('Number of episodes: '))
            train(episodes, agent, type(agent.learner).__name__)
            break
        elif typeGame == 1:
            print('0 - Q-Learning')
            print('1 - DeepQ-Learning')
            print('2 - SARSA')
            typeLearning = int(input('Select a type of pre-trained model: '))
            while True:
                if typeLearning == 0:
                    agent = FlappyAgent(learner=QLearning())
                    break
                elif typeLearning == 1:
                    agent = FlappyAgent(learner=DeepQLearning(8, 2))
                    break
                elif typeLearning == 2:
                    agent = FlappyAgent(learner=SARSA())
                    break
                else:
                    typeLearning = int(input('Please, insert a supported type: '))
            path_to_model = input('Relative path to the model: ')
            episodes = int(input('Number of episodes: '))
            agent.load_model(path_to_model)
            test(episodes, agent)
            break
        else:
            typeGame = int(input('Please, insert a supported action: '))