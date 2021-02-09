import torch
from .base_optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, model_params, lr=1e-3):
        super(SGD, self).__init__(model_params, lr)
    
    @torch.no_grad()
    def step(self, lr=None):
        if lr: self.lr = lr
        for param in self.model_params:
            param.sub_(self.lr * param.grad)