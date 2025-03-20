import torch
import torch.nn as nn

def card_to_num(card_str):
    """
    Convert a card string (e.g., '8S', 'QH') to a tuple of (rank, suit).

    Args:
        card_str: String representation of a card (e.g., '8S', 'QH', 'TC', 'AD')
                 T=10, J=Jack, Q=Queen, K=King, A=Ace
                 C=Clubs, D=Diamonds, H=Hearts, S=Spades
                 None or empty string for no card

    Returns:
        tuple: (rank, suit) where:
               rank is 0-12 (0=2, 1=3, ..., 12=Ace)
               suit is 0-3 (0=clubs, 1=diamonds, 2=hearts, 3=spades)
               (-1, -1) for no card
    """
    if not card_str:
        return (-1, -1)

    # Handle rank
    rank_char = card_str[1]
    if rank_char == 'T':
        rank = 8  # 10
    elif rank_char == 'J':
        rank = 9  # Jack
    elif rank_char == 'Q':
        rank = 10  # Queen
    elif rank_char == 'K':
        rank = 11  # King
    elif rank_char == 'A':
        rank = 12  # Ace
    else:
        rank = int(rank_char) - 2  # 2-9

    # Handle suit
    suit_char = card_str[0]
    suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
    suit = suit_map[suit_char]

    return [rank, suit]


class PokerCardEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128):
        """
        Transformer-based Poker Card Encoder

        Args:
            d_model: Dimension of the embedding and transformer model
            nhead: Number of heads in the multi-head attention
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Hidden dimension in the feed-forward network
        """
        super().__init__()

        # Constants
        self.num_suits = 4  # clubs, diamonds, hearts, spades
        self.num_ranks = 13  # 2-10, J, Q, K, A
        self.d_model = d_model

        # Embeddings for cards
        self.suit_embedding = nn.Embedding(self.num_suits, d_model // 2)
        self.rank_embedding = nn.Embedding(self.num_ranks, d_model // 2)

        # Position embedding (7 positions: 2 hole cards + 5 community cards)
        self.position_embedding = nn.Embedding(7, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output projection
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, cards, card_mask=None):
        """
        Forward pass

        Args:
            cards: Tensor of shape [batch_size, 7, 2] where:
                  - First dimension is batch size
                  - Second dimension is card position (0,1 for hole cards, 2-6 for community)
                  - Third dimension is [rank, suit] where:
                    - rank is 0-12 (0=2, 1=3, ..., 12=Ace)
                    - suit is 0-3 (0=clubs, 1=diamonds, 2=hearts, 3=spades)
                    - A value of -1 indicates no card (masked)
            card_mask: Boolean mask of shape [batch_size, 7] where:
                      - True means the card is not present
                      - False means the card is present

        Returns:
            encoded_cards: Tensor of shape [batch_size, 7, d_model]
        """
        batch_size, num_cards, _ = cards.shape

        # Extract ranks and suits
        ranks = cards[:, :, 0]  # [batch_size, 7]
        suits = cards[:, :, 1]  # [batch_size, 7]

        # Create mask if not provided
        if card_mask is None:
            card_mask = (ranks < 0) | (suits < 0)  # [batch_size, 7]

        # Replace -1 with 0 to avoid embedding errors
        ranks = torch.clamp(ranks, min=0)
        suits = torch.clamp(suits, min=0)

        # Get embeddings
        rank_emb = self.rank_embedding(ranks)  # [batch_size, 7, d_model//2]
        suit_emb = self.suit_embedding(suits)  # [batch_size, 7, d_model//2]

        # Combine rank and suit embeddings
        card_emb = torch.cat([rank_emb, suit_emb], dim=-1)  # [batch_size, 7, d_model]

        # Add position embeddings
        positions = torch.arange(num_cards).unsqueeze(0).expand(batch_size, -1).to(cards.device)
        pos_emb = self.position_embedding(positions)  # [batch_size, 7, d_model]

        # Combine card and position embeddings
        x = card_emb + pos_emb  # [batch_size, 7, d_model]

        # Create attention mask for transformer (True means don't attend)
        attn_mask = card_mask.unsqueeze(1).expand(batch_size, num_cards, num_cards)
        attn_mask = attn_mask.reshape(batch_size * num_cards, num_cards)

        # Apply transformer encoder
        x = x.view(-1, num_cards, self.d_model)  # Ensure correct shape
        x = self.transformer_encoder(x, src_key_padding_mask=card_mask)

        # Apply output projection
        encoded_cards = self.out_projection(x)

        return encoded_cards