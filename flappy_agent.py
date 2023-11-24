# from q_learning.q_learning_optimized import QLearning


class FlappyAgent:
    def __init__(self, learner):
        self.learner = learner

    def reward_values(self):
        """returns the reward values used for training

        Note: These are only the rewards used for training.
        The rewards used for evaluating the agent will always be
        1 for passing through each pipe and 0 for all other state
        transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end):
        """this function is called during training on each step of the game where
        the state transition is going from state s1 with action a to state s2 and
        yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

        Unless a terminal state was reached, two subsequent calls to observe will be for
        subsequent steps in the same episode. That is, s1 in the second call will be s2
        from the first call.
        """
        return self.learner.observe(s1, a, r, s2, end)

    def training_policy(self, state):
        """Returns the index of the action that should be done in state while training the agent.
        Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

        training_policy is called once per frame in the game while training
        """
        return self.learner.training_policy(state)

    def policy(self, state):
        """Returns the index of the action that should be done in state when training is completed.
        Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

        policy is called once per frame in the game (30 times per second in real-time)
        and needs to be sufficiently fast to not slow down the game.
        """
        return self.learner.policy(state)

    def save_model(self):
        return self.learner.save_model()

    def load_model(self, file_name="target_model.pth"):
        return self.learner.load_model(file_name)


