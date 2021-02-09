from torch import nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hid_dim[0])
        self.lin2 = nn.Linear(hid_dim[0], hid_dim[1])
        self.lin3 = nn.Linear(hid_dim[1], output_dim)

    def forward(self, x_in):
        #Flattening
        x_in = x_in.view(x_in.shape[0], -1)
        return self.lin3(F.relu(self.lin2(F.relu(self.lin1(x_in)))))
