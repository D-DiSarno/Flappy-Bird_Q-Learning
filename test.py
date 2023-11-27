from ple.games.flappybird import FlappyBird
from ple import PLE
from tqdm import tqdm
from matplotlib import pyplot as plt


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
