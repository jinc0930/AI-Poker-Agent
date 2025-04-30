import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass

@dataclass
class Hyperparams:
    learning_rate: float = 0.0003
    gamma: float = 0.95
    lmbda: float = 0.95
    eps_clip: float = 0.2
    K_epoch: int = 6
    T_horizon: int = 100
    entropy_coeff: float = 0.01

class PPO(nn.Module):
    def __init__(self, filename=None, device=None):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(23, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, 3)
        self.fc_v = nn.Linear(64, 1)

        if filename is not None:
            self.filename = filename
            self.load(filename)
        else:
            self.filename = None

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

    def load(self, filename):
        """Load the model parameters and optimizer state"""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"No model found at {filename}")
            
    def save_full(self, filename: str):
        """Save the entire model (architecture + parameters)"""
        torch.save(self)
        return filename

    def load_full(filename):
        """Load the entire model"""
        model = torch.load(filename)
        model.eval()
        return model