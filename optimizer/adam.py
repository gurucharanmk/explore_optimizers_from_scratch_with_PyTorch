import torch
from .base_optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, model_params, lr=1e-3, betas=(0.9, 0.999)):
        super(Adam, self).__init__(model_params, lr, momentum=betas[0])
        self.beta_1, self.beta_2 = betas
        self.avg_grads = [torch.zeros_like(p) for p in self.model_params]
        self.avg_sqr_grads = [torch.zeros_like(p) for p in self.model_params]
    
    @torch.no_grad()
    def step(self, lr=None):
        if lr: self.lr = lr
        steps = 0
        epsilon = 1e-8
        for param, avg_grad, avg_sqr_grad in zip(self.model_params, self.avg_grads, self.avg_sqr_grads):
            steps += 1
            avg_grad.mul_(self.beta_1).add_(param.grad * (1 - self.beta_1))
            avg_sqr_grad.mul_(self.beta_2).add_(param.grad * param.grad * (1 - self.beta_2))
            avg_grad_bias_corrected = avg_grad.div(1 - self.beta_1 ** steps)
            avg_sqr_grad_bias_corrected = avg_sqr_grad.div(1 - self.beta_2 ** steps)
            param.sub_(self.lr * avg_grad_bias_corrected / (avg_sqr_grad_bias_corrected.sqrt().add(epsilon)))