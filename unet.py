import pytorch_lightning as pt
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import math
from einops import rearrange

class GLU(nn.Module):
    """
    Custom implementation of GLU since the paper assumes GLU won't reduce
    the dimension of tensor by 2.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
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

        for _ in range(self.depth):
            # Encoder with padding to ensure consistent shapes
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=(1, 1)),
                    GLU(),
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                    GLU(),
                )
            )

            # Decoder with padding
            self.decoder.insert(
                0,
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                    GLU(),
                    nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(1, 1)),
                    GLU(),
                )
            )

            out_channels = mid_channels
            in_channels = mid_channels
            mid_channels = min(int(growth * mid_channels), max_channels)

        self.final_fc = nn.Linear(
            config['unet']['input_mel_dim'], config['unet']['final_mel_dim']
        ) if config['unet']['final_mel_dim'] != config['unet']['input_mel_dim'] else None

    def forward(self, x):
        if x.dim() < 4:
            x = rearrange(x, 'b (c f) t -> b c f t', c=1)

        skip_connections = []

        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)

        # Decoder
        for decode in self.decoder:
            skip = skip_connections.pop()

            # Ensure the shapes match for skip connection
            #if skip.size() != x.size():
            #    diff_h = skip.size(2) - x.size(2)
            #    diff_w = skip.size(3) - x.size(3)
            #    x = nn.functional.pad(x, (0, diff_w, 0, diff_h))

            x = decode(x + skip)

        # Final fully connected layer
        if self.final_fc:
            b, c, f, t = x.shape
            x = rearrange(x, 'b c f t -> (b t) (c f)')
            x = self.final_fc(x)
            x = rearrange(x, '(b t) f -> b f t', b=b)
        else:
            x = rearrange(x, 'b c f t->b (c f) t')
            
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
            'kernel_size': (4, 4),
            'growth': 2,
            'rescale': False,
            'stride': (2, 2),
            'padding': (1, 1),  # Ensures input-output size consistency
            'reference': None,
            'feature_dim': 160,
            'attention': False,
            'causal': False,
            'n_layers': 3,
            'n_heads': 8,
            'input_mel_dim': 160,
            'final_mel_dim': 80
        }
    }
    model = UNetRadio2D(config)
    x = torch.randn(1, 160, 256)
    y = model(x)
    print(x.shape)
    print(y.shape) # (1, 80, 256) expected
