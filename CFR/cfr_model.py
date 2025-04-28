import torch
import torch.nn as nn
import torch.nn.functional as F

class CardEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # only rank and suit embeddings
        self.rank = nn.Embedding(13, dim)
        self.suit = nn.Embedding(4, dim)

    def forward(self, cards):
        # cards: (B, N) with values 0–51 or -1 for “no card”
        mask = (cards >= 0).unsqueeze(-1).float()
        # clamp negatives to zero so indexing is safe
        cards_clamped = cards.clamp(min=0)
        rank_idx = cards_clamped // 4
        suit_idx = cards_clamped % 4

        emb = self.rank(rank_idx) + self.suit(suit_idx)
        emb = emb * mask  # zero out “no card”
        # sum over the N cards
        return emb.sum(dim=1)  # (B, dim)

class DeepCFRModel(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        dim_street: int = 8,
        dim_bet: int = 32,
        n_actions: int = 3,
    ):
        super().__init__()
        self.card_emb = CardEmbedding(dim)
        self.street_emb = nn.Embedding(4, dim_street)
        self.betting = nn.Sequential(
            nn.Linear(5, dim_bet),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim_bet),
        )
        # two‐layer trunk with residual
        input_size = dim * 2 + dim_street + dim_bet
        self.trunk1 = nn.Linear(input_size, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.trunk2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        self.head = nn.Linear(dim, n_actions)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hole_cards, community_cards, street, betting_feats):
        # hole_cards: (B,2), community_cards: (B,5)
        h_emb = self.card_emb(hole_cards)            # (B, dim)
        c_emb = self.card_emb(community_cards)       # (B, dim)
        s_emb = self.street_emb(street)              # (B, dim_street)
        b_emb = self.betting(betting_feats)          # (B, dim_bet)

        x = torch.cat([h_emb, c_emb, s_emb, b_emb], dim=1)
        # first layer + residual
        y = F.relu(self.norm1(self.trunk1(x)))
        y = y + F.relu(x) if x.shape == y.shape else y

        # second layer + residual
        z = F.relu(self.norm2(self.trunk2(y)))
        z = z + y

        return self.head(z)  # (B, n_actions)
