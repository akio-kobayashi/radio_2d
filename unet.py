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
        self.kernel_size = config['unet']['kernel_size']  # (height, width)
        growth = config['unet']['growth']
        self.stride = config['unet']['stride']  # (height_stride, width_stride)
        self.padding = config['unet'].get('padding', (0, 0))  # 明示的にパディングを指定
        reference = config['unet']['reference']

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        feature_dim = config['unet']['feature_dim']

        for index in range(self.depth):
            encode = nn.ModuleList()
            encode.append(
                nn.Conv2d(
                    in_channels,
                    mid_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
            )
            encode.append(nn.ReLU())
            encode.append(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1)
            )
            encode.append(nn.ReLU())
            self.encoder.append(nn.Sequential(*encode))
            
            decode = nn.ModuleList()
            decode.append(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1)
            )
            decode.append(nn.ReLU())
            decode.append(
                nn.ConvTranspose2d(
                    mid_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
            )
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            
            out_channels = mid_channels
            in_channels = mid_channels
            mid_channels = min(int(growth * mid_channels), max_channels)

        self.attention = None
        if config['unet']['attention']:
            # TFEncoder: channels * feature_dim を d_model として渡す
            self.attention = TFEncoder(in_channels * feature_dim, 
                                       n_layers=config['unet']['n_layers'], 
                                       n_heads=config['unet']['n_heads'])

        self.final_fc = None
        if config['unet']['final_mel_dim'] != config['unet']['input_mel_dim']:
            self.final_fc = nn.Linear(config['unet']['final_mel_dim'], config['unet']['input_mel_dim'])

    def forward(self, x):
        # x: (batch, feature, time)
        if x.dim() < 3:
            x = rearrange(x, 'b (c f) t -> b c f t', c=1) 
        # x: (batch, channels, feature, time)
        skip_connections = []

        # Encoder pass
        for encode in self.encoder:
            x = encode(x)  # 2Dエンコード
            skip_connections.append(x)

        # Attention module
        if self.attention:
            b, c, f, t = x.shape
            x = rearrange(x, 'b c f t -> b t (c f)')
            #x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (batch, time, d_model)
            x = self.attention(x)  # TFEncoder expects (batch, seq_len, d_model)
            x = rearrange(x, 'b t (c f) -> b c f t')
            #x = x.view(b, t, c, f).permute(0, 2, 3, 1)  # (batch, channels, feature, time)

        # Decoder pass
        for decode in self.decoder:
            skip = skip_connections.pop()
            x = decode(x + skip)  # Skip connection and decode

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
