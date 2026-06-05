import torch
from copy import deepcopy

class EarlyStopping:
    def __init__(self, patience, min_delta, model):
        self.patience, self.min_delta, self.model = patience, min_delta, model
        self.best_value, self.best_state, self.counter, self.best_epoch = float('inf'), None, 0, 0


    def _improved(self, v):
        return v < (self.best_value - self.min_delta)


    def step(self, value, epoch):
        if self._improved(value):
            self.best_value = value
            self.best_epoch = epoch
            self.best_state = deepcopy(self.model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience





