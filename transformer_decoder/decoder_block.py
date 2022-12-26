from multiheadattentiom import *
from transformerblock import *
from utils import *

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=4):
        super(DecoderBlock, self).__init__()

        self.attention = MultiheadedAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, query, value, mask):
        attention = self.attention(key, query, value, mask)