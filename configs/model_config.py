from dataclasses import dataclass

@dataclass
class ModelConfig:
    nbins: int
    hidden_size: int
    num_transformer_layers: int
    num_heads: int
    num_groups: int
    dropout: float
    lr_size: int
    target_size: int
    latent_dim: int
    apply_sht: bool