import numpy as np
import random
import os

# Q-learning parameters

ALPHA = 0.01
GAMMA = 0.8
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 1000
NUM_ACTIONS = 2

class QLearningAgent:
    def __init__(self, gm, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY):
        self.num_games = 0
        self.Q_table = {}  # Q-table to store Q values
        self.gm = gm
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def new_gm(self, gm):
        self.gm = gm

    def get_state(self):

     return tuple(self.gm.getState())

    def get_action(self, state, epsilon=True):
        self.epsilon = np.interp(self.num_games, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

        #if epsilon and random.uniform(0, 1) < self.epsilon:
        #    return random.choice(range(NUM_ACTIONS))
        #else:
          #  state = self.get_state()
          #  if state not in self.Q_table:
         #       return random.choice(range(NUM_ACTIONS))
          #  else:
           #     return max(range(NUM_ACTIONS), key=lambda a: self.Q_table[state][a])
        return [0,1]

    def remember(self, state, action, reward, new_state, done):
        state = tuple(state)
        new_state = tuple(new_state)

        if state not in self.Q_table:
            self.Q_table[state] = [0] * NUM_ACTIONS
        if new_state not in self.Q_table:
            self.Q_table[new_state] = [0] * NUM_ACTIONS

        action_index = np.argmax(action)  # Convert action to an integer index
        max_future_q = max(self.Q_table[new_state])
        current_q = self.Q_table[state][action_index]

        if not done:
            new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_future_q)
        else:
            new_q = (1 - ALPHA) * current_q + ALPHA * reward

        self.Q_table[state][action_index] = new_q

    def train_long_memory(self):
        pass  # Q-learning without experience replay doesn't need this

    def train_short_memory(self, state, action, reward, new_state, done):
        self.remember(state, action, reward, new_state, done)

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



        agent.train_short_memory(state, action, reward, new_state, done)

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
                print(f"Game: {agent.num_games}, Score: {score}, Epsilon: {agent.epsilon}")
                print(f"State: {state}")
                # Plotting function here

        if epochs == agent.num_games:
            agent.save_scores(record, total_score, run_num, 'train')
            break
