import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gym_dataset import GymDataset

from gym_canvas import CanvasEnv
from model import FCN

class RendererTrainer:

    def __init__(self, env, model):
        self.device = torch.device("cuda")
        self.lr = 3e-6

        self.env = env
        self.dataset = GymDataset(env)
        
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        train_loader = DataLoader(self.dataset)
        
        self.model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            print(data.shape, target.shape)
            continue

            output = self.model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def valid(self):
        pass

    def visualize(self):
        pass

def main():
    parser = argparse.ArgumentParser(description='Renderer Training')

    args = parser.parse_args()

    model = FCN()
    env = CanvasEnv()

    trainer = RendererTrainer(env, model)
    trainer.train()

if __name__ == "__main__":
    main()
