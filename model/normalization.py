import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
    

class DualEarNormalization(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.left_norm = RMSNorm(dim // 2, eps)
        self.right_norm = RMSNorm(dim // 2, eps)

    def forward(self, x):
        left = self.left_norm(x[..., :x.shape[-1] // 2])
        right = self.right_norm(x[..., x.shape[-1] // 2:])
        return torch.cat([left, right], dim=-1)


if __name__ == "__main__":
    x = torch.randn(1, 138, 128)
    norm = RMSNorm(128)
    x = norm(x)
    print(x.shape)

    x = torch.randn(2, 18, 256)
    dual_ear_norm = DualEarNormalization(256)
    x = dual_ear_norm(x)
    print(x.shape)