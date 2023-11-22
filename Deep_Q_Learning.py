from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from helper import plot

# Parametri di Q-learning
ALPHA = 0.001
GAMMA = 0.8
BATCH_SIZE = 32
BUFFER_SIZE = 700000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 1000


class Model(nn.Module):
    def __init__(self, gm):
        super().__init__()
        self.linear1 = nn.Linear(len(gm.getState()), 24)
        self.linear2 = nn.Linear(24, 12)
        self.linear3 = nn.Linear(12, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

    def save(self, run_num, type, file_name='best_weights.pt'):
        if run_num is None:
            model_folder_path = './model/' + str(type)
        else:
            model_folder_path = './model/' + str(type) + str(run_num)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)

        torch.save(self.state_dict(), file_name)


class ModelTrainer:
    def __init__(self, model, alpha=ALPHA, gamma=GAMMA):
        self.alpha = alpha
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.alpha)

    def train_step(self, state, action, reward, new_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        new_state = torch.tensor(np.array(new_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.get_q_values(new_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = F.mse_loss(pred, target.detach())
        loss.backward()
        self.optimizer.step()

    def get_q_values(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float)
        return self.model(state)


class Agent:
    def __init__(self, gm, model=Model, model_trainer=ModelTrainer, epsilon_start=EPSILON_START,
                 epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE):
        self.num_games = 0
        self.memory = deque(maxlen=buffer_size)
        self.gm = gm
        self.model = model(self.gm)
        self.trainer = model_trainer(self.model)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def new_gm(self, gm):
        self.gm = gm

    def get_state(self):
        return np.array(self.gm.getState(), dtype=int)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, new_state, done):
        self.trainer.train_step(state, action, reward, new_state, done)

    def get_action(self, state, epsilon=True):
        self.epsilon = np.interp(self.num_games, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
        action = [0, 0]
        if epsilon:
            if self.epsilon >= random.uniform(0.0, 1.0):
                move = random.randint(0, 1)
                action[move] = 1
            else:
                state_t = torch.tensor(state, dtype=torch.float)
                predict = self.model(state_t)
                move = torch.argmax(predict).item()
                action[move] = 1
        else:
            state_t = torch.tensor(state, dtype=torch.float)
            predict = self.model(state_t)
            move = torch.argmax(predict).item()
            action[move] = 1
            self.gm.setOutputs(predict)

        return action

    def save_scores(self, record, total_score, run_num, type, file_name='scores.txt'):
        if run_num is None:
            model_folder_path = './model/' + str(type)
        else:
            model_folder_path = './model/' + str(type) + str(run_num)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        with open(file_name, 'w') as r:
            r.write(str(record) + '\n' + str(total_score / self.num_games))


def train(agent, run_num=None, epochs=None, plotting_scores=False):
    total_score = 0
    record = 0
    agent = agent
    plot_scores = []
    plot_mean_scores = []

    agent.gm.reset()
    agent.num_games = 0

    while True:
        state = agent.get_state()
        action = agent.get_action(state)
        reward, done, score = agent.gm.actionSequence(action)
        new_state = agent.get_state()
        agent.remember(state, action, reward, new_state, done)

        if done:
            total_score += score
            agent.gm.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(run_num, 'train')
                agent.save_model(record, total_score, run_num, 'train')

            if plotting_scores:
                plot_scores.append(score)
                mean_score = total_score / agent.num_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores, "Training...", agent.num_games)

        if epochs == agent.num_games:
            agent.model.save(run_num, 'train')
            agent.save_model(record, total_score, run_num, 'train')
            break


def evaluate(agent, run_num=None, epochs=None, plotting_scores=False):
    total_score = 0
    record = 0
    agent = agent
    plot_scores = []
    plot_mean_scores = []

    agent.gm.reset()
    agent.num_games = 0

    while True:
        state = agent.get_state()
        action = agent.get_action(state, epsilon=False)
        _, done, score = agent.gm.actionSequence(action)

        if done:
            total_score += score
            agent.gm.reset()
            agent.num_games += 1

            if score > record:
                record = score
                agent.save_model(record, total_score, run_num, 'evaluate')

            if plotting_scores:
                plot_scores.append(score)
                mean_score = total_score / agent.num_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores, "Evaluating...", agent.num_games)

        if epochs == agent.num_games:
            agent.save_model(record, total_score, run_num, 'evaluate')
            break
