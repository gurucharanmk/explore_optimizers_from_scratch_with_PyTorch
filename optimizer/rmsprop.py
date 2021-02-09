import torch
from .base_optimizer import Optimizer

class RMSProp(Optimizer):
    def __init__(self, model_params, lr=1e-3, momentum=0.9, alpha=0.99):
        super(RMSProp, self).__init__(model_params, lr, momentum)
        self.alpha = alpha
        self.avg_sqr_grads = [torch.zeros_like(p) for p in self.model_params]
    
    @torch.no_grad()
    def step(self, lr=None):
        if lr: self.lr = lr
        epsilon = 1e-8
        for param, avg_sqr_grad in zip(self.model_params, self.avg_sqr_grads):
            avg_sqr_grad.mul_(self.alpha).add_(param.grad * param.grad * (1 - self.alpha))
            std = avg_sqr_grad.sqrt().add(epsilon)
            param.sub_((self.lr / std) * param.grad)