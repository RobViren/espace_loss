import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class MSSLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x, y):
        loss = 0.0
        for n_fft, hop, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.ones(win_length, device=x.device)
            
            x_stft = torch.stft(x.squeeze(1), n_fft, hop, win_length, window=window, return_complex=True, center=False)
            y_stft = torch.stft(y.squeeze(1), n_fft, hop, win_length, window=window, return_complex=True, center=False)
            
            x_mag = x_stft.abs() + 1e-7
            y_mag = y_stft.abs() + 1e-7
            loss += F.l1_loss(x_mag, y_mag)
            loss += F.l1_loss(x_mag.log(), y_mag.log())
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return F.mse_loss(pred, target)
