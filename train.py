from matplotlib import pyplot as plt
from helper import plot as plot
from ple.games.flappybird import FlappyBird
from ple import PLE
from tqdm import tqdm
import os


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

    model_dir = f"model_{model_name}_{str(total)}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    agent.save_model(filename=os.path.join(model_dir, f"q_table_{str(total)}"))

    plt.plot(game_scores, label="Score per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(os.path.join(model_dir, f"plot_{str(total)}.png"))
    plt.show()
