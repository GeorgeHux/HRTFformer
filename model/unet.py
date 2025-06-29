import torch
import torch.nn as nn
from configs.model_config import ModelConfig
from .common import Reshape, Trim

class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.transConv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.PReLU()
    
    def forward(self, x):
        x = self.transConv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Unet(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        initial_size = model_config.initial_size
        in_channels = 256
        self.fc = nn.Sequential(
            nn.Linear(model_config.latent_dim, initial_size*in_channels),
            nn.BatchNorm1d(initial_size * in_channels),
            nn.PReLU(),
            Reshape(-1, in_channels, initial_size)
        )

        self.layers = nn.ModuleList()
        out_channels = [256, 256, 256, 256, 256]
        for channels in out_channels:
            self.layers.append(TransConvBlock(in_channels, channels))
            in_channels = channels

        self.layers.append(Trim(model_config.target_size, dim=1))
        self.out_conv = nn.Conv1d(in_channels, model_config.nbins, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.fc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out_conv(x)
        return x
