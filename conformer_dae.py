import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.nn import Transformer
from einops import rearrange

class ConformerBlock(nn.Module):
    """Conformer Block"""
    def __init__(self, dim, num_heads, conv_kernel_size=31):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.layer_norm2 = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, kernel_size=1),
            nn.GLU(dim=1),  # Gated Linear Unit
            nn.Conv1d(dim, dim, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),  # Swish Activation
            nn.Conv1d(dim, dim, kernel_size=1)
        )
        
        self.layer_norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        """x: (batch, seq_len, dim)"""
        # Self Attention
        x = x + self.self_attn(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))[0]
        
        # Convolution Module
        x = x + self.conv(self.layer_norm2(x).transpose(1, 2)).transpose(1, 2)
        
        # Feed Forward Network
        x = x + self.ffn(self.layer_norm3(x))
        
        return x

class DenoisingConformerDAE(nn.Module):
    """Conformer-based Denoising Autoencoder"""
    def __init__(self, input_dim=80, seq_len=256, d_model=256, num_heads=8, num_layers=6, conv_kernel_size=31):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # メルスペクトルを埋め込み次元に変換
        self.pos_encoding = nn.Parameter(torch.rand(1, seq_len, d_model))  # 位置エンコーディング

        self.conformer_blocks = nn.Sequential(
            *[ConformerBlock(d_model, num_heads, conv_kernel_size) for _ in range(num_layers)]
        )

        self.output_layer = nn.Linear(d_model, input_dim)  # 埋め込み次元 → メルスペクトログラム次元

    def forward(self, noisy_mel):
        batch_size, mel_dim, num_frames = noisy_mel.shape  # (B, 80, 256)

        # (B, 80, 256) → (B, 256, 80) に転置
        x = rearrange(noisy_mel, 'b m t -> b t m') #noisy_mel.permute(0, 2, 1)

        # メル次元を Transformer の埋め込み次元 (d_model) に変換
        x = self.embedding(x)  # (B, 256, d_model)

        # 位置エンコーディングを加算
        x = x + self.pos_encoding

        # Conformer Blocks
        x = self.conformer_blocks(x)  # (B, 256, d_model)

        # (B, 256, d_model) → (B, 256, 80)
        clean_mel = self.output_layer(x)

        # 元の形状に戻す (B, 80, 256)
        clean_mel = rearrange(clean_mel, 'b t m -> b m t') #clean_mel.permute(0, 2, 1)
        return clean_mel
    
class DenoisingTransformerDAE(nn.Module):
    def __init__(self, input_dim=80, seq_len=256, d_model=256, num_heads=8, num_layers=6):
        super(DenoisingTransformerDAE, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)  # メルスペクトル次元 → 埋め込み次元
        self.pos_encoding = nn.Parameter(torch.rand(1, seq_len, d_model))  # 位置エンコーディング

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=512),
            num_layers=num_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=512),
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(d_model, input_dim)  # 埋め込み次元 → メルスペクトル次元

    def forward(self, noisy_mel):
        batch_size, mel_dim, num_frames = noisy_mel.shape  # (B, 80, 256)

        # (B, 80, 256) → (B, 256, 80) に転置
        x = rearrange(noisy_mel, 'b m t -> b t m') #noisy_mel.permute(0, 2, 1)  # 時間方向をシーケンスとする

        # メル次元を Transformer の埋め込み次元 (d_model) に変換
        x = self.embedding(x)  # (B, 256, d_model)

        # 位置エンコーディングを加算
        x = x + self.pos_encoding  # (B, 256, d_model)

        # Transformer Encoder-Decoder
        x = self.encoder(x)  # (B, 256, d_model)
        x = self.decoder(x, x)  # (B, 256, d_model)

        # (B, 256, d_model) → (B, 256, 80)
        clean_mel = self.output_layer(x)

        # 元の形状に戻す (B, 80, 256)
        clean_mel = rearrange(clean_mel, 'b t m -> b m t') #clean_mel.permute(0, 2, 1)
        return clean_mel

if __name__ == '__main__':
    config = {
        'conformer': {
            'input_dim': 80,
            'seq_len': 256,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
            'conv_kernel_size': 31
        },
        'transformer':{
            'input_dim': 80,
            'seq_len': 256,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
        }
    }    
    import numpy as np
    np.random.seed(0)
    x = torch.randn(1, 80, 256)
    model = DenoisingConformerDAE(**config['conformer'])
    y = model(x)
    


