import torch
from .base_optimizer import Optimizer

class SGDWithMomentum(Optimizer):
    def __init__(self, model_params, lr=1e-3, momentum=0.9):
        super(SGDWithMomentum, self).__init__(model_params, lr, momentum)
        self.v = [torch.zeros_like(p) for p in self.model_params]
    
    @torch.no_grad()
    def step(self, lr=None):
        if lr: self.lr = lr
        for param, v in zip(self.model_params, self.v):
            v.mul_(self.mom).add_(param.grad)
            param.sub_(self.lr * v)