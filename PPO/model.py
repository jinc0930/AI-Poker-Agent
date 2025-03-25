import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass
import os

@dataclass
class Hyperparams:
    learning_rate: float = 0.0001 # Lower LR to improve stability
    gamma: float = 0.99 # long-term strategic planning
    lmbda: float = 0.95
    eps_clip: float = 0.2 # Allows more policy exploration
    K_epoch: int = 10
    T_horizon: int = 100

class PPO(nn.Module):
    def __init__(self, filename=None, device=None, hyperparams: Hyperparams = Hyperparams()):
        super(PPO, self).__init__()
        self.hyperparams = hyperparams
        self.data = []

        # Set device to CUDA if available or provided device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fc1 = nn.Linear(43, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_pi = nn.Linear(128, 3)
        self.fc_v = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=hyperparams.learning_rate)

        # Move model to device
        self.to(self.device)
        if filename is not None:
            self.load(filename)
            self.filename = filename
        else:
            self.filename = None

    def pi(self, x, softmax_dim=0):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def reward(self, r=1):
        if len(self.data) > 0:
            x = self.data[-1]
            y = list(x)
            y[2] = r
            self.data[-1] = tuple(y)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s = torch.tensor(s_lst, dtype=torch.float).to(self.device)
        a = torch.tensor(a_lst).to(self.device)
        r = torch.tensor(r_lst).to(self.device)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float).to(self.device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(self.device)
        prob_a = torch.tensor(prob_a_lst).to(self.device)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self, force = False):
        if len(self.data) < self.hyperparams.T_horizon and not force:
            return
        if len(self.data) <= 8:
            return

        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(self.hyperparams.K_epoch):
            td_target = r + self.hyperparams.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()  # Move to CPU for numpy operations

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.hyperparams.gamma * self.hyperparams.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.hyperparams.eps_clip, 1+self.hyperparams.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def save(self, filename = None):
        """Save the model parameters and optimizer state"""
        state = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state, self.filename if filename is None else filename)
        # print(f"Model saved to {filename}")
        return filename

    def load(self, filename):
        """Load the model parameters and optimizer state"""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # print(f"Model loaded from {filename}")

            # Fix optimizer device if needed
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        else:
            # print(f"No model found at {filename}")
            pass