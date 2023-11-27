import torch
from models.deep_q_learning import DeepQLearning
from models.q_learning import QLearning
from models.sarsa import SARSA
from agent import FlappyAgent
from train import train
from test import test

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # agent = FlappyAgent(learner=DeepQLearning(8, 2))
    agent = FlappyAgent(learner=QLearning())
    # agent = FlappyAgent(learner=SARSA())

    train(1000, agent, type(agent.learner).__name__)
    # agent.load_model("model_100/q_table_100.pth")
    # test(50, agent)
