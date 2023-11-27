import random
import math
import json
from .utils import Q_Parameters, _ACTIONS, _FLAP, _NO_FLAP, _MIN_V, _MAX_V, _MAX_X, _MAX_Y

#SARSA aggiorna q-valori con valori stato prossimo e azione dello stato attuale
class SARSA:
    def __init__(self):
        self.eps_start = Q_Parameters["epsilon_start"]
        self.eps_decay = Q_Parameters["epsilon_decay"]
        self.eps_end = Q_Parameters["epsilon_end"]
        self.alpha = Q_Parameters["alpha"]
        self.alpha_end = Q_Parameters["alpha_end"]
        self.alpha_decay = Q_Parameters["alpha_decay"]
        self.gamma = Q_Parameters["gamma"]
        self.partitions = Q_Parameters["partitions"]
        self.steps_done = Q_Parameters["steps_done"]

        self._ACTIONS = _ACTIONS
        self._FLAP = _FLAP
        self._NO_FLAP = _NO_FLAP
        self._MIN_V = _MIN_V
        self._MAX_V = _MAX_V
        self._MAX_X = _MAX_X
        self._MAX_Y = _MAX_Y

        self.Q_table = {
            ((y_pos, pipe_top_y, x_dist, velocity), action): 0
            for y_pos in range(self.partitions)
            for pipe_top_y in range(self.partitions)
            for x_dist in range(self.partitions)
            for velocity in range(-8, 11)
            for action in range(2)
        }

    def observe(self, s1, a, r, s2, end):
        s1 = self._state_encoder(s1)
        s2 = self._state_encoder(s2)

        future_reward = self.Q_table[(s2, r)] if not end else 0

        old_alpha = self.alpha
        self.alpha = old_alpha * self.alpha_decay

        self.Q_table[(s1, a)] = self.Q_table[(s1, a)] + max(
            self.alpha_end, old_alpha
        ) * (r + self.gamma * future_reward - self.Q_table[(s1, a)])

    def training_policy(self, state):
        state = self._state_encoder(state)
        threshold = random.uniform(0, 1)

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if threshold < eps_threshold:
            return random.choice(self._ACTIONS)
        elif self.Q_table[(state, self._FLAP)] == self.Q_table[(state, self._NO_FLAP)]:
            return random.choice(self._ACTIONS)
        elif self.Q_table[(state, self._FLAP)] > self.Q_table[(state, self._NO_FLAP)]:
            return self._FLAP
        else:
            return self._NO_FLAP

    def policy(self, state):
        state = self._state_encoder(state)

        if self.Q_table[(state, self._FLAP)] == self.Q_table[(state, self._NO_FLAP)]:
            return random.choice(self._ACTIONS)
        elif self.Q_table[(state, self._FLAP)] > self.Q_table[(state, self._NO_FLAP)]:
            return self._FLAP
        else:
            return self._NO_FLAP

    def save_model(self, filename="q_table.json"):
        q_table_str_keys = {str(key): value for key, value in self.Q_table.items()}

        with open(f"{filename}.json", "w") as file:
            json.dump(q_table_str_keys, file)

    def load_model(self, filename="q_table.json"):
        with open(filename, "r") as file:
            q_table_str_keys = json.load(file)

        self.Q_table = {eval(key): value for key, value in q_table_str_keys.items()}

    def _state_encoder(self, state):
        y_pos = max(0, min(state["player_y"], 512))
        pipe_top_y = max(25, min(state["next_pipe_top_y"], 192))
        x_dist = max(0, min(state["next_pipe_dist_to_player"], 288))
        velocity = max(-8, min(state["player_vel"], 10))

        enc_y_pos = self._get_interval(512, self.partitions, y_pos)
        enc_pipe_top_y = self._get_interval(512, self.partitions, pipe_top_y)
        enc_x_dist = self._get_interval(288, self.partitions, x_dist)

        return enc_y_pos, enc_pipe_top_y, enc_x_dist, velocity

    def _get_interval(self, total_size, interval, value):
        interval_width = total_size / interval
        partition = value // interval_width

        if partition < 0:
            return 0
        if partition > interval - 1:
            return interval - 1

        return partition
