import torch
# from models.deep_q_learning import DeepQLearning
from models.q_learning import QLearning
from flappy_agent import FlappyAgent
from train import train
from test import test

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # agent = FlappyAgent(learner=DeepQLearning(8,2))
    agent = FlappyAgent(learner=QLearning())

    train(10000, agent)
    # agent.load_model("q_table30000.json")
    # test(50, agent)
