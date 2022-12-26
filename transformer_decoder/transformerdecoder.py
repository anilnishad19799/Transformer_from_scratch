from positional_embedding import *
from utils import *
from decoder_block import *

class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()

        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbeding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            DecoderBlock(embed_dim, expansion_factor=4, n_heads=8) for _ in range(num_layers)
        )

        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

        def forward(self, x, enc_out, trg_mask):

            batch_size, seq_length = x.shape[0], x.shape[1]

            x = self.word_embedding(x)
            x = self.positional_embedding(x)
            x = self.dropout(x)

            for layer in self.layers:
                x = layer(enc_out, enc_out, x, trg_mask)

            out = F.softmax(self.fc_out(x))

            return out