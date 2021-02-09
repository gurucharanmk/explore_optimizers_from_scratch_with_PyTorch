import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, train_dl, model, optimizer, criterion, epochs=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_dl = train_dl
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.losses = {}

    def init_params(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
    def one_batch(self, X, Y):
        self.optimizer.zero_grad()
        y = self.model(X)
        loss = self.criterion(y, Y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def one_epoch(self, epoch):
        opt_name = str(self.optimizer.__class__.__name__)
        with tqdm(self.train_dl, unit="batch") as tepoch:
            for X, Y in tepoch:
                tepoch.set_description(f"{opt_name}(epoch:{epoch})")
                X = X.to(self.device)
                Y = Y.to(self.device)
                loss = self.one_batch(X, Y)
                self.losses[opt_name].append(loss)
                tepoch.set_postfix(loss=loss)
    
    def train(self):
        opt_name = str(self.optimizer.__class__.__name__)
        self.losses[opt_name] = []
        self.init_params()
        for epoch in range(self.epochs):
            self.one_epoch(epoch)
            
    def analyze_optimizers(self, optimizers):
        for optimizer in optimizers:
            self.optimizer = optimizer
            self.train()
    
    def plot_loss(self, scale=None):
        _, ax = plt.subplots(figsize=(15,5))
        for label, loss in self.losses.items():
            ax.plot(loss, label=label)
        ax.set_ylabel('Loss')
        ax.set_title('Loss plot')
        ax.set_xlabel('Batch count')
        ax.set_ylim(ymin=0, ymax=None)
        ax.grid()
        ax.legend(loc='upper right')