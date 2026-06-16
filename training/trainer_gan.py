import torch
import torch.nn as nn

from .checkpoint import save_checkpoint, load_checkpoint
from .losses_v1 import GANLoss, mae, psnr

def to_01(x: torch.Tensor):
    if x.min().item() < 0:
        x = x * 0.5 + 0.5

    return x.clamp(0, 1)

class GANTrainer(nn.Module):
    def __init__(self, cfg, model, train_loader, 
                 optimizers: dict, valid_loader = None, gan_mode: str = 'lsgan', save_dir: str = 'checkpoints')