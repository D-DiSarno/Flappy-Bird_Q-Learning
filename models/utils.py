# costanti gioco
_FLAP, _NO_FLAP = 0, 1
_MIN_V, _MAX_V = -8, 10
_MAX_X, _MAX_Y = 288, 512
_ACTIONS = [_FLAP, _NO_FLAP]

# Parametri per il Q
Q_Parameters = {
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "alpha": 0.25,
    "alpha_end": 0.1,
    "gamma": 0.99,
    "partitions": 30,
    "epsilon_decay": 100000,
    "alpha_decay": 0.999,
    "steps_done": 0,
}

# Parametri per il DQN
DQN_Parameters = {
    "batch_size": 128,
    "gamma": 0.99,
    "eps_start": 0.9,
    "eps_end": 0.01,
    "eps_decay": 100000,
    "tau": 0.005,
    "lr": 1e-4 * 5,
    "update_freq": 100,
    "steps_done": 0,
    "update_counter": 0,
}

# Valori reward
REWARD_Values = {
    "positive": 1.0,
    "tick": 0.0,
    "loss": -10.0,
}