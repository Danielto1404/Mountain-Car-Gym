from collections import defaultdict

import torch


class Selector:
    def choose_action(self, state):
        raise NotImplementedError


class ActionSelector(Selector):
    def __init__(self, model, device=None):
        super(ActionSelector, self).__init__()
        self.model = model
        self.device = device

    @torch.no_grad()
    def choose_action(self, state):
        """
        :return: (best action, Q-value for best action)
        """
        tensor_state = torch.tensor(state, device=self.device, dtype=torch.float32)
        q_values = self.model(tensor_state)
        action = torch.argmax(q_values).item()
        return action, q_values[action].item()


class EnsembleSelector(Selector):
    def __init__(self, models, device=None):
        super(EnsembleSelector, self).__init__()
        self.selectors = [ActionSelector(model=m, device=device) for m in models]
        self.device = device

    @torch.no_grad()
    def choose_action(self, state):
        counter = defaultdict(lambda: (0, 0))
        actions_ensemble = [s.choose_action(state) for s in self.selectors]

        for (action, q_value) in actions_ensemble:
            amount, q = counter[action]
            counter[action] = (amount + 1, q + q_value)

        best_action = max(counter, key=lambda k: counter[k])
        amount, q_max = counter.get(best_action)
        return best_action, q_max / amount


def load_selector(path) -> Selector:
    if isinstance(path, list):
        return EnsembleSelector([torch.load(p) for p in path])
    else:
        return ActionSelector(torch.load(path))
