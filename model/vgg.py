import torch
import torch.nn as nn
from configs.model_config import ModelConfig
from .common import initial_size_to_strides_map

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.PReLU()
        )
    
    def forward(self, x):
        x = self.features(x)
        return x

class VGGNet(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        initial_size = model_config.initial_size

        # strides for downsampling layers
        self.strides = initial_size_to_strides_map[initial_size]
        in_channels = model_config.nbins
        num_encoding_layer = len(self.strides)
        self.layers = nn.ModuleList()
        for i in range(num_encoding_layer):
            out_channels = min(in_channels * 2, 2048)
            self.layers.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels

        self.fc = nn.Sequential(nn.Linear(initial_size * in_channels, 1024),
                                nn.BatchNorm1d(1024),
                                nn.PReLU(),
                                nn.Linear(1024, model_config.latent_dim))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x