import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from .position_embedding import RotaryEmbedding

class GroupedQueryAttention(nn.Module):
    def __init__(self, emb_size, hidden_size, num_heads, num_groups, dropout=0., max_num_coeff=484):
        super(GroupedQueryAttention, self).__init__()
        self.hidden_size = hidden_size
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query_proj = nn.Linear(emb_size, hidden_size*num_groups)
        self.key_proj = nn.Linear(emb_size, hidden_size)
        self.value_proj = nn.Linear(emb_size, hidden_size)

        # rotary position embedding
        self.query_rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_num_coeff)
        self.key_rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_num_coeff)

        self.out_proj = nn.Linear(hidden_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        b, q_len = query.shape[:2]
        query = self.query_proj(query) # [b, q_len, hidden_size]
        query = rearrange(query, "b sq (n d) -> b n sq d", d=self.head_dim)
        query = self.query_rope(query)
        query = rearrange(query, "b (g h) sq d -> b g h sq d", g=self.num_groups)
        key = self.key_proj(key)
        key = rearrange(key, "b sk (h d) -> b h sk d", d=self.head_dim)
        key = self.key_rope(key)
        value = self.value_proj(value)
        value = rearrange(value, "b sv (h d) -> b h sv d", d=self.head_dim)
        scale = self.head_dim ** 0.5
        # calculate attention scores
        scores = einsum(query, key, "b g h sq d, b h sk d -> b h sq sk")
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(scores / scale, dim=3) # [b, head_per_group, q_len, k_len]

        out = einsum(attention, value, "b h sq sk, b h sv d -> b sq h d").reshape(b, q_len, -1) # sk = sv
        out = self.out_proj(out) # [b, q_len, emb_size]
        return out

if __name__ == "__main__":
    batch_size = 2
    query_len = 84
    key_len = 120
    embed_dim = 256
    query = torch.randn(batch_size, query_len, embed_dim)
    key = torch.randn(batch_size, key_len, embed_dim)
    value = torch.randn(batch_size, key_len, embed_dim)
    hidden_size = 1024
    num_heads = 16
    num_groups = 4
    gqa = GroupedQueryAttention(embed_dim, hidden_size, num_heads, num_groups)
    out = gqa(query, key, value)
    print(out.shape)





