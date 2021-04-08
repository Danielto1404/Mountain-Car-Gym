import numpy as np


class EpsilonStrategy:
    def __init__(self, start=1, decay=5e-3, min_eps=0.01):
        self.eps = start
        self.decay = decay
        self.min_eps = min_eps

    def eps(self):
        return self.eps

    def decrease(self):
        self.eps = max(self.eps - self.decay, self.min_eps)

    def check_random_prob(self):
        return np.random.random(1) < self.eps
