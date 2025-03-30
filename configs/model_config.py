from dataclasses import dataclass

@dataclass
class ModelConfig:
    nbins: int
    hidden_size: int
    num_transformer_layers: int
    num_heads: int
    num_groups: int
    dropout: float
    num_initial_coeff: int
    max_num_coeff: int
    latent_dim: int