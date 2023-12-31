import torch
import torch.optim as optim
import math
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple, deque
from .utils import DQN_Parameters, _ACTIONS, _FLAP, _NO_FLAP,_MIN_V, _MAX_V,_MAX_X, _MAX_Y,REWARD_Values

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "end")
)


class ReplayMemory(object):
    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DeepQLearning:

    def __init__(
            self,
            n_observations,
            n_actions,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.device = device

        self.batch_size = DQN_Parameters["batch_size"]
        self.gamma = DQN_Parameters["gamma"]
        self.eps_start = DQN_Parameters["eps_start"]
        self.eps_end = DQN_Parameters["eps_end"]
        self.eps_decay = DQN_Parameters["eps_decay"]
        self.tau = DQN_Parameters["tau"]
        self.lr = DQN_Parameters["lr"]

        self.steps_done = DQN_Parameters["steps_done"]
        self.episode_durations = []

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )

        self.memory = ReplayMemory(10000)
        self.N = DQN_Parameters["update_freq"]
        self.update_counter = DQN_Parameters["update_counter"]

        self._ACTIONS = _ACTIONS
        self._FLAP = _FLAP
        self._NO_FLAP = _NO_FLAP
        self._MIN_V = _MIN_V
        self._MAX_V = _MAX_V
        self._MAX_X = _MAX_X
        self._MAX_Y = _MAX_Y


    def reward_values(self):

        positve=REWARD_Values["positive"]
        tick=REWARD_Values["tick"]
        loss=REWARD_Values["loss"]
        return {positve,tick,loss}

    def observe(self, s1, a, r, s2, end):
        enc_s1 = self._state_encoder(s1)
        enc_s2 = self._state_encoder(s2) if s2 is not None else None
        self._store_transition(enc_s1, a, r, enc_s2, end)
        self._optimize_model()

    def training_policy(self, state):
        return self._select_action(state, training=True)

    def policy(self, state):
        return self._select_action(state, training=False)

    def _store_transition(self, s1, a, r, s2, end):

        s1 = torch.tensor([s1], device=self.device, dtype=torch.float32)
        a = torch.tensor([[a]], device=self.device, dtype=torch.long)
        r = torch.tensor([r], device=self.device, dtype=torch.float32)
        end = torch.tensor([end], device=self.device, dtype=torch.bool)


        s2 = (
            torch.tensor([s2], device=self.device, dtype=torch.float32)
            if not end
            else None
        )


        self.memory.push(s1, a, r, s2, end)

    def _select_action(self, state, training=False):

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if training and sample < eps_threshold:

            return random.choice(self._ACTIONS)
        else:

            with torch.no_grad():
                encoded_state = self._state_encoder(state)
                state_tensor = torch.tensor(
                    [encoded_state], device=self.device, dtype=torch.float32
                )

                q_values = self.policy_net(state_tensor)

                return q_values.max(1)[1].item()

    def _optimize_model(self):

        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))


        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)


        state_action_values = self.policy_net(state_batch).gather(1, action_batch)


        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch


        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))


        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if self.update_counter % self.N == 0:
            self._soft_update()
        self.update_counter += 1

    def _soft_update(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def _state_encoder(self, state):
        y_pos = max(0, min(state["player_y"], 512))
        pipe_top_y = max(25, min(state["next_pipe_top_y"], 192))
        pipe_bottom_y = max(125, min(state["next_pipe_bottom_y"], 292))
        next_pipe_top_y = max(25, min(state["next_next_pipe_top_y"], 192))
        next_pipe_bottom_y = max(125, min(state["next_next_pipe_bottom_y"], 292))
        x_dist = max(0, min(state["next_pipe_dist_to_player"], 288))
        next_x_dist = max(0, min(state["next_next_pipe_dist_to_player"], 288))
        velocity = max(-8, min(state["player_vel"], 10))

        enc_y_pos = (y_pos) / (self._MAX_Y)
        enc_pipe_top_y = pipe_top_y / self._MAX_Y
        enc_pipe_bottom_y = pipe_bottom_y / self._MAX_Y
        enc_x_dist = x_dist / self._MAX_X
        next_enc_pipe_top_y = next_pipe_top_y / self._MAX_Y
        next_enc_pipe_bottom_y = next_pipe_bottom_y / self._MAX_Y
        next_enc_x_dist = next_x_dist / self._MAX_X
        normalized_velocity = (velocity - self._MIN_V) / (self._MAX_V - self._MIN_V)

        return (
            enc_y_pos,
            enc_pipe_top_y,
            enc_pipe_bottom_y,
            enc_x_dist,
            next_enc_pipe_top_y,
            next_enc_pipe_bottom_y,
            next_enc_x_dist,
            normalized_velocity,
        )

    def save_model(self, filename="target_model.pth"):
        torch.save(self.target_net.state_dict(), f"{filename}.pth")

    def load_model(self, file_name="target_model.pth"):
        self.target_net.load_state_dict(torch.load(file_name))
        self.target_net.eval()
        print(f"Model loaded from {file_name}.")
