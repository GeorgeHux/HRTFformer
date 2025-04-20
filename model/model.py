import torch
import torch.nn as nn
import math
from .transformer import Encoder as TransformerLayer
from configs.model_config import ModelConfig

num_initial_coeff_to_stides_map = {
    25: [2, 2, 2],
    16: [2, 2, 1],
    9: [2, 1, 1],
    4: [1, 1, 1],
}

lr_size_to_strides_map = {
    27: [2, 2, 2],
    25: [2, 2, 2],
    18: [2, 2, 1],
    16: [2, 2, 1],
    9: [2, 1, 1],
    8: [2, 1, 1],
    5: [1, 1, 1],
    4: [1, 1, 1],
    3: [1, 1, 1]
}

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args
    
    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x[:,:self.shape,...]

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DownsampleLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # input shape: [batch_size, num_elements (coefficients or raw hrtf points), channels]
        x = x.permute(0, 2, 1) # adjust to [batch_size, channels, num_elements]
        x = self.conv(x)
        x = x.permute(0, 2, 1) # adjust back to [batch_size, num_elements, channels]
        x = self.gelu(x)
        x = self.norm(x)
        return x

class Encoder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(Encoder, self).__init__()
        assert model_config.lr_size in lr_size_to_strides_map, f"invalid initial lr size, should be one of {lr_size_to_strides_map.keys()}"

        # strides for downsampling layers
        self.strides = lr_size_to_strides_map[model_config.lr_size]
        in_channels = model_config.nbins
        # each layer of Encoder model is constructed by a transformer layer followed by a downsampling layer
        # except the last layer, which is only a transformer layer without downsampling
        # for example, if total number encoding layer is 5, the structure is as:
        # [Transofrmer, downsampling, transformer, downsampling, transformer, downsampling, transformer]
        # strides only indicate the stride used in each downsampling layer
        # therefore the total number of encoding layer is len(strides) + 1
        num_encoding_layer = len(self.strides) + 1
        self.layers = nn.ModuleList()
        for i in range(len(self.strides) + 1):
            self.layers.append(TransformerLayer(emb_size=in_channels,
                                                hidden_size=model_config.hidden_size,
                                                num_layers=model_config.num_transformer_layers,
                                                num_heads=model_config.num_heads,
                                                num_groups=model_config.num_groups,
                                                dropout=model_config.dropout,
                                                target_size=model_config.target_size))
            # no downsampling for last layer
            if i < num_encoding_layer - 1:
                self.layers.append(DownsampleLayer(in_channels=in_channels, out_channels=in_channels*2,
                                                   stride=self.strides[i])) # downsamply by 2 if stride=2
            in_channels *= 2
        
        output_size = self._get_output_dim(model_config.lr_size)
        self.fc = nn.Sequential(nn.Linear(output_size * in_channels // 2, 1024),
                                nn.BatchNorm1d(1024),
                                # nn.PReLU(),
                                nn.GELU(),
                                nn.Linear(1024, model_config.latent_dim))
        self.latent_conv = nn.Sequential(
            nn.Conv1d(in_channels // 2, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1024),
            # nn.PReLU(),
            nn.GELU(),
            nn.Conv1d(1024, model_config.latent_dim, kernel_size=3, stride=1, padding=1)
        )

    def _get_output_dim(self, lr_size):
        size = lr_size
        # configuration for convolution layer
        kernel_size = 3
        padding = 1
        # compute the output shape
        for s in self.strides:
            size = (size + 2 * padding - kernel_size) // s + 1
        return size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # x = x.permute(0, 2, 1)
        # x = self.latent_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class TrimLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x

class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(UpsampleLayer, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=stride, stride=stride)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # input shape: [batch_size, num_elements (coefficients or raw hrtf points), channels]
        x = x.permute(0, 2, 1) # adjust to [batch_size, channels, num_elements]
        x = self.conv_transpose(x)
        x = x.permute(0, 2, 1) # adjust back to [batch_size, num_elements, channels]
        x = self.gelu(x)
        x = self.norm(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(Decoder, self).__init__()
        in_channels = 1024
        self.fc = nn.Sequential(
            nn.Linear(model_config.latent_dim, 4*in_channels),
            nn.BatchNorm1d(4 * in_channels),
            # nn.PReLU(),
            nn.GELU(),
            Reshape(-1, 4, in_channels)
        )
        self.conv0 = nn.Conv1d(model_config.latent_dim, in_channels, kernel_size=3, stride=1, padding=1)
        self.layers = nn.ModuleList()
        if model_config.apply_sht:
            # for SH coefficients: 4->8->16->32->64->128->256->512
            out_channels = [1024, 1024, 512, 512, 256, 256, 256]
        else:
            # for raw hrtf points: 4->8->16->32->64->128->256->512->1024
            out_channels = [1024, 1024, 512, 512, 512, 256, 256, 256]
        num_layers = len(out_channels) + 1

        for layer_index in range(num_layers):
            self.layers.append(TransformerLayer(emb_size=in_channels,
                                                hidden_size=model_config.hidden_size,
                                                num_layers=model_config.num_transformer_layers,
                                                num_heads=model_config.num_heads,
                                                num_groups=model_config.num_groups,
                                                dropout=model_config.dropout,
                                                target_size=model_config.target_size))
            if layer_index < num_layers - 1:
                self.layers.append(UpsampleLayer(in_channels=in_channels,out_channels=out_channels[layer_index]))
                in_channels = out_channels[layer_index]
            if layer_index == num_layers - 2:
                self.layers.append(Trim(model_config.target_size))
    
    def forward(self, x):
        # x = self.conv0(x)
        # x = x.permute(0, 2, 1)
        x = self.fc(x)
        for layer in self.layers:
            x = layer(x)
        return x

class HRTF_Transformer(nn.Module):
    def __init__(self, encoder_config, decoder_config) -> None:
        super(HRTF_Transformer, self).__init__()
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)

    def forward(self, x):
        encoder_out = self.encoder(x)
        sr = self.decoder(encoder_out)
        return sr.permute(0, 2, 1)
