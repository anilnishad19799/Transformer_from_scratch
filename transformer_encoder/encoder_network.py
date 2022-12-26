from transformerblock import *
from utils import *
from embedding import *
from positional_embedding import *

class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=2):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = Embedding(vocab_size, seq_len)
        self.positional_encoder = PositionalEmbeding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])

        def forward(self, x):
            embed_out = self.embedding_layer(x)
            out = self.positional_encoder(embed_out)
            for layer in self.layers:
                out = layer(out, out, out)

            return out