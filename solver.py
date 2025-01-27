import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple
from einops import rearrange
from model import Generator, Discriminator, AuxDiscriminator
import math
from unet import UNetRadio2D

class CustomLRScheduler(object):
    def __init__(self, optimizer, n_samples, lr, epochs, mini_batch_size) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.n_samples = n_samples
        self.lr = lr
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.lr_decay = self.lr / float(self.epochs * (self.n_samples // self.mini_batch_size))
        self.current_step = 0
        
    def step(self):
        self.lr = max(0., self.lr - self.lr_decay)
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] = self.lr
        self.current_step += 1
        
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'lr': self.lr,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.lr = state_dict['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
            
'''
 PyTorch Lightning 用 solver
'''
class LitDAE(pl.LightningModule):
    def __init__(self, config:dict) -> None:        
        super().__init__()
        self.config = config
        self.model = UNetRadio2D(config)

    def forward(self, data:Tensor) -> Tensor:
        return self.model(data) 
   
    def compute_loss(self, pred_clean, real_clean, valid=False):
        d = {}
        _loss = 0.

        # L1 loss
        if valid is True:
            d['valid_loss'] = _loss
        else:
            d['loss'] = _loss

        self.log_dict(d)

        return _loss

    def training_step(self, batch, batch_idx:int) -> Tensor:
        self.model.train()
        real_clean, _, real_noisy, mask_noisy = batch
        pred_clean = self.model(real_noisy * mask_noisy)
        _loss = self.compute_loss(pred_clean, real_clean, valid=False)

        # return nothing becuase of manual updates   
        return _loss

    def validation_step(self, batch, batch_idx: int):
        self.model.eval()
        # Forward pass for Generator
        with torch.no_grad():  # Disable gradient computation for validation
            real_clean, _, real_noisy, mask_noisy = batch
            pred_clean = self.model(real_noisy * mask_noisy)
            _loss = self.compute_loss(pred_clean, real_clean, valid=True)

        # Return losses for further analysis
        return _loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.config['optimizer'])
        return (
            {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.config['process']['batch_size']),
                "monitor": "val_loss"
                }
            }
        )
        
