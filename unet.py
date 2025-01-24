import pytorch_lightning as pt
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import math
from einops import rearrange

class TFEncoder(nn.Module):
    def __init__(self, dim, n_layers=5, n_heads=8):
        super(TFEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=n_heads,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=n_layers)

    def forward(self, x, mask=None, padding_mask=None):
        return self.encoder(x, mask, padding_mask)

class UNetRadio2D(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.depth = config['unet']['depth']

        in_channels = config['unet']['in_channels']
        mid_channels = config['unet']['mid_channels']
        out_channels = config['unet']['out_channels']
        max_channels = config['unet']['max_channels']
        kernel_size = config['unet']['kernel_size']
        stride = config['unet']['stride']
        growth = config['unet']['growth']

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(self.depth):
            # Calculate padding to ensure 2^n shape
            padding = self.calculate_padding(kernel_size, stride)

            # Encoder
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.ReLU(),
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                )
            )

            # Decoder
            self.decoder.insert(
                0,
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.ReLU() if index > 0 else nn.Identity(),
                )
            )

            in_channels = mid_channels
            mid_channels = min(int(growth * mid_channels), max_channels)

        self.attention = None
        if config['unet']['attention']:
            self.attention = TFEncoder(
                dim=in_channels * config['unet']['feature_dim'], 
                n_layers=config['unet']['n_layers'], 
                n_heads=config['unet']['n_heads']
            )

        self.final_fc = nn.Linear(config['unet']['final_mel_dim'], config['unet']['input_mel_dim']) if config['unet']['final_mel_dim'] != config['unet']['input_mel_dim'] else None

    @staticmethod
    def calculate_padding(kernel_size, stride):
        """
        Calculate padding to ensure output dimensions are divisible by stride.
        """
        padding_h = (stride[0] - 1 + kernel_size[0]) // 2
        padding_w = (stride[1] - 1 + kernel_size[1]) // 2
        return (padding_h, padding_w)

    def forward(self, x):
        # Adjust input shape to (batch, channels, features, time)
        if x.dim() < 4:
            x = rearrange(x, 'b (c f) t -> b c f t', c=1)
        
        skip_connections = []

        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)

        # Attention
        if self.attention:
            b, c, f, t = x.shape
            x = rearrange(x, 'b c f t -> b t (c f)')
            x = self.attention(x)
            x = rearrange(x, 'b t (c f) -> b c f t')

        # Decoder
        for decode in self.decoder:
            skip = skip_connections.pop()
            x = decode(x + skip)

        # Final fully connected layer
        if self.final_fc:
            x = rearrange(x, 'b c f t -> b c t f')
            x = self.final_fc(x)
            x = rearrange(x, 'b c t f -> b c f t')

        return x
    
if __name__ == '__main__':
    # Config for UNetRadio2D
    config = {
        'unet': {
            'normalize': False,
            'floor': False,
            'resample': False,
            'depth': 3,  # Number of encoder-decoder layers
            'in_channels': 1,
            'mid_channels': 16,
            'out_channels': 1,
            'max_channels': 64,
            'kernel_size': (3, 3),
            'growth': 2,
            'rescale': False,
            'stride': (2, 2),
            'padding': (1, 1),  # Ensures input-output size consistency
            'reference': None,
            'feature_dim': 80,
            'attention': False,
            'causal': False,
            'n_layers': 3,
            'n_heads': 8,
            'input_mel_dim': 256,
            'final_mel_dim': 80
        }
    }
    model = UNetRadio2D(config)
    x = torch.randn(1, 256, 256)
    y = model(x)
    print(x.shape)
    print(y.shape) # (1, 80, 256) expected
