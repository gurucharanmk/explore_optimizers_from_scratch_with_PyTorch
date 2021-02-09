from abc import ABC,abstractmethod 
class Optimizer(ABC):
    def __init__(self, model_params, lr=1e-3, momentum=0.9):
        self.model_params = list(model_params)
        self.lr = lr
        self.mom = momentum

    def zero_grad(self):
        for param in self.model_params:
            param.grad = None

    @abstractmethod
    def step(self, lr=None):
        pass