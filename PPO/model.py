import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from encoder import PokerCardEncoder

#Hyperparameters
learning_rate = 0.01
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20
num_actions   = 3
other_features = 3

class PPO(nn.Module):
    def __init__(self, d_model=64):
        super(PPO, self).__init__()
        self.data = []

        # Card encoder using transformer
        self.card_encoder = PokerCardEncoder(d_model=d_model)

        # Feature size after pooling the card embeddings (we'll use mean pooling)
        feature_size = d_model

        # Combined feature size
        combined_size = feature_size + other_features + 1

        self.fc1   = nn.Linear(combined_size,256)
        self.fc_pi = nn.Linear(256,num_actions)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def encode_cards(self, cards, card_mask=None):
        """
        Encode the poker cards using the transformer encoder

        Args:
            cards: Tensor of shape [batch_size, 7, 2]
            card_mask: Boolean mask of shape [batch_size, 7]

        Returns:
            encoded_features: Tensor of shape [batch_size, d_model]
        """
        # Get card embeddings from transformer [batch_size, 7, d_model]
        encoded_cards = self.card_encoder(cards, card_mask)

        # Mean pooling across cards (ignoring masked cards)
        if card_mask is not None:
            # Create a mask for averaging (1 for valid cards, 0 for masked cards)
            # Shape: [batch_size, 7, 1]
            avg_mask = (~card_mask).float().unsqueeze(-1)

            # Apply mask and compute mean
            # Shape: [batch_size, d_model]
            encoded_features = (encoded_cards * avg_mask).sum(dim=1) / avg_mask.sum(dim=1)
        else:
            # Simple mean if no mask
            encoded_features = encoded_cards.mean(dim=1)

        return encoded_features

    def put_data(self, transition):
        self.data.append(transition)

    def reward(self, r = 1):
        if (len(self.data) > 0):
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

        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()