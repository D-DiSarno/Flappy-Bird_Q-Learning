class FlappyAgent:
    def __init__(self, learner):
        self.learner = learner

    def reward_values(self):

        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end):

        return self.learner.observe(s1, a, r, s2, end)

    def training_policy(self, state):

        return self.learner.training_policy(state)

    def policy(self, state):

        return self.learner.policy(state)

    def save_model(self, filename="q_table.json"):
        return self.learner.save_model(filename=filename)

    def load_model(self, file_name="target_model.pth"):
        return self.learner.load_model(file_name)
