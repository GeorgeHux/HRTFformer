import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    # x1_rot = x1 * cos - x2 * sin
    # x2_rot = x1 * sin + x2 * cos
    # x = [x1, x2]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        """
        Rotary Embedding initialization

        Args:
            dim (int): input feature dimension
            max_seq_len (int): max number of SH coefficients
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        t = torch.arange(max_seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    
    def forward(self, x):
        seq_len = x.shape[-2]
        cos = self.cos_cached[:,:,:seq_len, ...]
        sin = self.sin_cached[:,:,:seq_len, ...]
        return apply_rotary_pos_emb(x, cos, sin)

if __name__ == "__main__":
    dim = 256
    x = torch.randn(2, 32, 122, dim)
    rotaryEmbedding = RotaryEmbedding(dim)
    x_embed = rotaryEmbedding(x)
    print(x_embed.shape)