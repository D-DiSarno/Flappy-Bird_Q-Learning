import matplotlib.pyplot as plt
from IPython import display

plt.ion()
path_to_model = ""


def plot(scores, mean_scores, type_game, num_games):
    if num_games % 50 == 0:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title(type_game)
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
        plt.savefig(path_to_model + "plot.png")
        plt.show(block=False)

    plt.pause(.1)
