from utils import *
from multiheadattentiom import *

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, exapansion_factor=4, n_heads=4):
        super(TransformerBlock, self).__init__()

        self.attention = MultiheadedAttention(embed_dim, n_heads)
        self.norm1 = nn.Linear(embed_dim)
        self.norm2 = nn.Linear(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, exapansion_factor*embed_dim),
            nn.ReLU(),
            nn.Linear(exapansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        def forward(self, key, query, value, mask=None):

            attention_out = self.attention(key, query, value, mask)
            attention_residual_out =  attention_out + value
            norm1_out = self.dropout1(self.norm1(attention_residual_out))

            feed_fwd_out = self.feed_forward(norm1_out)

            feed_fwd_residual_out = feed_fwd_out + norm1_out

            norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

            return norm2_out