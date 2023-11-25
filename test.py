from ple.games.flappybird import FlappyBird
from ple import PLE
from tqdm import tqdm
from matplotlib import pyplot as plt
def test(nb_episodes, agent):
    """Runs nb_episodes episodes of the game with agent picking the moves.
    An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {
        "positive": 1.0,
        "negative": 0.0,
        "tick": 0.0,
        "loss": 0.0,
        "win": 0.0,
    }
    # TODO: when training use the following instead:
    #reward_values = agent.reward_values

    env = PLE(
        FlappyBird(),
        fps=30,
        display_screen=False,
        force_fps=True,
        rng= None ,
        reward_values=reward_values,
    )
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True
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
            #action = agent.training_policy(env.game.getGameState())
            # step the environment
            reward = env.act(env.getActionSet()[action])
            print("reward=%d" % reward)

            # TODO: for training let the agent observe the current state transition

            score += reward

            # reset the environment if the game is over
            if env.game_over():
                if score > 1000 and score > max_score:
                    agent.save_model()
                    max_score = score

                if nb_episodes % 20 == 0 and scores != []:
                    print(
                        f"Max score for {total - nb_episodes} episodes: {max(scores)}"
                    )
                    print(
                        f"Avg score for {total - nb_episodes} episodes: {sum(scores) / len(scores)}"
                    )

                pbar.update(1)
                scores.append(score)
                env.reset_game()
                nb_episodes -= 1
                score = 0
    print(f"90% over 50: {check_ninety_percent_over_fifty(scores)}")
    print(f"Max Test score: {max(scores)}")

def check_ninety_percent_over_fifty(array):
    total_elements = len(array)
    count_over_fifty = sum(1 for element in array if element >= 50)
    percentage_over_fifty = (count_over_fifty / total_elements) * 100
    return percentage_over_fifty >= 90

