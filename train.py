import matplotlib.pyplot as plt
from ple.games.flappybird import FlappyBird
from ple import PLE
from tqdm import tqdm


def train(nb_episodes, agent):
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

    frames = 0
    total_rewards = []
    scores = []
    total = nb_episodes

    score = 0
    with tqdm(total=nb_episodes, position=0, leave=True) as pbar:
        while nb_episodes > 0:
            # pick an action
            state = env.game.getGameState()
            action = agent.training_policy(state)

            # step the environment
            reward = env.act(env.getActionSet()[action])
            if reward > 10:
                print("reward=%d" % reward)

            # let the agent observe the current state transition
            newState = env.game.getGameState()
            agent.observe(state, action, reward, newState, env.game_over())
            frames += 1

            score += reward

            # reset the environment if the game is over
            if env.game_over():
                if nb_episodes % 1000 == 0 and scores != []:
                    print(
                        f" [MAX: {max(scores)}] [AVG: {sum(scores) / len(scores)}]"
                    )
                    scores = []
                if score > 1000:
                    break
                scores.append(score)
                total_rewards.append(score)
                env.reset_game()
                env.reset_game()
                pbar.update(1)
                nb_episodes -= 1
                score = 0
        print(f"Max Train score: {max(scores)}")
        print(f"Avg score for {total} episodes: {sum(scores) / len(scores)}")

    plt.plot(total_rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    print(f"Number of Frames: {frames}")
