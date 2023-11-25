from ple.games.flappybird import FlappyBird
from ple import PLE
from tqdm import tqdm
import torch
# from models.q_learning import QLearning
from models.deep_q_learning import DeepQLearning
from models.q_learning_optimized import QLearning
from flappy_agent import FlappyAgent
from train import train


def run_game(nb_episodes, agent):
    """Runs nb_episodes episodes of the game with agent picking the moves.
    An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = agent.reward_values()

    env = PLE(
        FlappyBird(),
        fps=30,
        display_screen=True,
        force_fps=False,
        rng=None,
        reward_values=reward_values,
    )
    env.init()
    scores = []
    total = nb_episodes

    max_score = float("-inf")
    score = 0
    with tqdm(total=nb_episodes, position=0, leave=True) as pbar:
        while nb_episodes > 0:
            # pick an action
            # TODO: for training using agent.training_policy instead
            action = agent.policy(env.game.getGameState())
            # action = agent.training_policy(env.game.getGameState())
            # step the environment
            reward = env.act(env.getActionSet()[action])

            # TODO: for training let the agent observe the current state transition

            score += reward

            # reset the environment if the game is over
            if env.game_over():
                if score > 1000 and score > max_score:
                    # agent.save_model()
                    max_score = score

                if nb_episodes % 20 == 0 and scores != []:
                    print(
                        f" [MAX: {max(scores)}] [AVG: {sum(scores) / len(scores)}]"
                    )

                pbar.update(1)
                scores.append(score)
                env.reset_game()
                nb_episodes -= 1
                score = 0
    print(f"\n90% over 50: {check_ninety_percent_over_fifty(scores)}")
    print(f"Max Test score: {max(scores)}")


def check_ninety_percent_over_fifty(array):
    total_elements = len(array)
    count_over_fifty = sum(1 for element in array if element >= 50)
    percentage_over_fifty = (count_over_fifty / total_elements) * 100
    return percentage_over_fifty >= 90


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # agent = FlappyAgent(learner=DeepQLearning(8,2))
    agent = FlappyAgent(learner=QLearning())

    # train(30000, agent)
    agent.load_model("q_table30000.json")
    run_game(100, agent)
