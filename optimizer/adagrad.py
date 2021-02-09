import torch
from .base_optimizer import Optimizer

class Adagrad(Optimizer):
    def __init__(self, model_params, lr=1e-3, momentum=0.9):
        super(Adagrad, self).__init__(model_params, lr, momentum)
        self.acc_sqr_grads = [torch.zeros_like(p) for p in self.model_params]
    
    @torch.no_grad()
    def step(self, lr=None):
        if lr: self.lr = lr
        epsilon = 1e-8
        for param, acc_sqr_grad in zip(self.model_params, self.acc_sqr_grads):
            acc_sqr_grad.add_(param.grad * param.grad)
            std = acc_sqr_grad.sqrt().add(epsilon)
            param.sub_((self.lr / std) * param.grad)