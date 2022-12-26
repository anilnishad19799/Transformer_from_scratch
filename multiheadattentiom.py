from utils import *

class MultiheadedAttention(nn.Module):
    def __init__(self, embed_size, n_heads=8):
        super(MultiheadedAttention, self).__init__()

        self.embed_dim = embed_size
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)

        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)

        self.out = nn.Linear(self.n_heads*self.single_head_dim, self.embed_dim)

    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)

        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(value)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        k_adjusted = k.transpose(-1,-2)
        product = torch.matmul(q, k_adjusted)

        if mask is not None:
            product = product / math.sqrt(self.single_head_dim)
        
        score  = F.softmax(product, dims=-1)

        scores = torch.matmul(scores, v)

        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length, self.single_head_dim*self.n_heads)

        output = self.out(concat)

        return output        