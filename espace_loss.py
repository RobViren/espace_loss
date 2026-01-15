import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

class BidirectionalCursedIIR(nn.Module):
    def __init__(self, sr=24000, n_channels=64, f_min=10, f_max=100, r_min=0.9999, r_max=0.99999):
        super().__init__()
        freqs = torch.exp(torch.linspace(np.log(f_min), np.log(f_max), n_channels))
        radii = torch.linspace(r_min, r_max, n_channels)
        
        theta = 2 * np.pi * freqs / sr
        a0 = torch.ones_like(radii)
        a1 = -2 * radii * torch.cos(theta)
        a2 = radii ** 2
        
        # This normalizes the peak gain to roughly 1.0
        b0 = (1 - radii) * 0.5 
        
        self.register_buffer('a_coeffs', torch.stack([a0, a1, a2], dim=1).double())
        self.register_buffer('b_coeffs', torch.stack([b0, torch.zeros_like(b0), torch.zeros_like(b0)], dim=1).double())

    def run_filter(self, x):
        x_expanded = x.repeat(1, self.a_coeffs.shape[0], 1).double()
        return torchaudio.functional.lfilter(x_expanded, self.a_coeffs, self.b_coeffs, clamp=False).float()

    def forward(self, x):
        fwd_state = self.run_filter(x)
        x_flipped = torch.flip(x, dims=[-1])
        bwd_state = torch.flip(self.run_filter(x_flipped), dims=[-1])
        return torch.cat([fwd_state, bwd_state], dim=1)

class Tank(nn.Module):
    def __init__(self, in_dim=128, out_dim=64, seed=1337):
        super().__init__()
        torch.manual_seed(seed)
        self.mix = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.mix.weight)

    def forward(self, x):
        z = self.mix(x.transpose(1, 2))
        return torch.tanh(z).transpose(1, 2)

class ESpaceLoss(nn.Module):
    def __init__(self, device='cpu', sr=24000, res_size=64, align_weight=10.0, raw_weight=0.1):
        super().__init__()
        self.align_weight = align_weight 
        self.raw_weight = raw_weight
        
        self.probe = BidirectionalCursedIIR(sr, n_channels=res_size).to(device)
        self.tank = Tank(in_dim=res_size*2, out_dim=res_size).to(device)
        
        # Freeze
        for p in self.probe.parameters(): p.requires_grad = False
        for p in self.tank.parameters(): p.requires_grad = False
        
        self.fft_sizes = [512, 1024, 2048]
        self.hop_sizes = [128, 256, 512]
        self.win_lengths = [512, 1024, 2048]
        
    def mss_loss(self, x, y):
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            x_stft = torch.stft(x.squeeze(1), n_fft, hop, win, return_complex=True, center=False)
            y_stft = torch.stft(y.squeeze(1), n_fft, hop, win, return_complex=True, center=False)
            x_mag = x_stft.abs() + 1e-7
            y_mag = y_stft.abs() + 1e-7
            loss += F.l1_loss(x_mag, y_mag)
            loss += F.l1_loss(x_mag.log(), y_mag.log())
        return loss

    def forward(self, pred, target):
        T = pred.shape[-1]
        margin = T // 3
        
        l_mss = self.mss_loss(pred, target)
        
        p_emb = self.tank(self.probe(pred))
        t_emb = self.tank(self.probe(target))
        
        # Slice center
        p_valid = p_emb[..., margin:-margin]
        t_valid = t_emb[..., margin:-margin]
        
        # L1 on State Space (should now be ~0.2 to ~0.8)
        l_align = F.l1_loss(p_valid, t_valid)
        
        # Tiny raw L1 hole-puncher for the zero-crossing barrier
        l_raw = F.l1_loss(pred[..., margin:-margin], target[..., margin:-margin]) * self.raw_weight
        
        total_loss = l_mss + (l_align * self.align_weight) + l_raw
        
        return total_loss, l_mss.item(), l_align.item()