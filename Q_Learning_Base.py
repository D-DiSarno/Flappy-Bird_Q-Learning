from collections import deque
import random
import numpy as np
import os
import torch
from datetime import datetime
from helper import plot

ALPHA = 0.01
GAMMA = 2.5
BATCH_SIZE = 64
BUFFER_SIZE = 700000
EPSILON_START = 1.0
EPSILON_END = .02
EPSILON_DECAY = 7000


class QLearningAgent:
    def __init__(self, gm, alpha=ALPHA, gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                 epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE):
        self.num_games = 0
        self.gamma = gamma
        self.memory = deque(maxlen=buffer_size)
        self.gm = gm
        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = {}  # Aggiunta dell'inizializzazione della tabella Q

    def new_gm(self, gm):
        self.gm = gm

    def get_state(self):
        return np.array(self.gm.getState(), dtype=int)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def train_step(self, state, action, reward, new_state, done):
        state_discrete = self.discretize_state(state)
        new_state_discrete = self.discretize_state(new_state)

        if state_discrete not in self.q_table:
            self.q_table[state_discrete] = [0, 0]

        if new_state_discrete not in self.q_table:
            self.q_table[new_state_discrete] = [0, 0]

        current_q = self.q_table[state_discrete][action]
        max_future_q = np.max(self.q_table[new_state_discrete])

        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[state_discrete][action] = new_q

    def discretize_state(self, state):
        binary_str = "".join(map(lambda s: str(int(s)), state))
        try:
            return int(binary_str, 2)
        except ValueError:
            return 0  # o un altro valore di default se la conversione fallisce

    def get_action(self, state, epsilon=True):
        self.epsilon = np.interp(self.num_games, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
        action = [0, 0]

        if epsilon and self.epsilon >= random.uniform(0.0, 1.0):
            move = random.randint(0, 1)
            action[move] = 0
        else:
            state_discrete = self.discretize_state(state)
            if state_discrete not in self.q_table:
                self.q_table[state_discrete] = [0, 0]

            move = np.argmax(self.q_table[state_discrete])
            action[move] = 1

        return action

    def save_scores(self, record, total_score, run_num, type, file_name='scores.txt'):
        if run_num is None:
            model_folder_path = './models/model' + datetime.today().strftime('_%Y-%m-%d_%H-%M') + '/' + str(type)
        else:
            model_folder_path = './models/model' + datetime.today().strftime('_%Y-%m-%d_%H-%M') + '/' + str(type) + str(run_num)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        with open(file_name, 'w') as r:
            r.write(str(record) + '\n' + str(total_score / self.num_games))


def train(agent, run_num=None, epochs=None, plotting_scores=False):
    total_score = 0
    record = 0
    plot_scores = []
    plot_mean_scores = []

    # resets the environment
    agent.gm.reset()
    agent.num_games = 0

    while True:
        # get old state
        state = agent.get_state()
        # get move
        action = agent.get_action(state)
        # action then new state
        reward, done, score = agent.gm.actionSequence(action)
        new_state = agent.get_state()
        agent.remember(state, action, reward, new_state, done)

        if done:
            total_score += score
            agent.gm.reset()
            agent.num_games += 1

            if score > record:
                record = score
                agent.save_scores(record, total_score, run_num, 'train')

            if plotting_scores:
                plot_scores.append(score)
                mean_score = total_score / agent.num_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores, "Training...", agent.num_games)

        if epochs == agent.num_games:
            agent.save_scores(record, total_score, run_num, 'train')
            break


def evaluate(agent, run_num=None, epochs=None, plotting_scores=False):
    total_score = 0
    record = 0
    plot_scores = []
    plot_mean_scores = []

    # resets the environment
    agent.gm.reset()
    agent.num_games = 0

    while True:
        # get old state
        state = agent.get_state()
        # get move
        action = agent.get_action(state, epsilon=False)
        # action then new state
        _, done, score = agent.gm.actionSequence(action)

        if done:
            total_score += score
            agent.gm.reset()
            agent.num_games += 1

            if score > record:
                record = score
                agent.save_scores(record, total_score, run_num, 'evaluate')

            if plotting_scores:
                plot_scores.append(score)
                mean_score = total_score / agent.num_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores, "Evaluating...", agent.num_games)

        if epochs == agent.num_games:
            agent.save_scores(record, total_score, run_num, 'evaluate')
            break
