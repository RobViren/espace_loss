import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np


class Probe(nn.Module):
    """
    A bidirectional IIR filter bank that acts as holographic ESN-like embedding layer projecting audio through a fixed distribution into a high dimensional temporal space giving the embedding both future and past context.
    """
    def __init__(self, n_channels=64, r_min=0.9999, r_max=0.99999, theta_min=None, theta_max=None):
        super().__init__()
        
        if theta_min is None:
            theta_min = 2 * np.pi * 10.0 / 24000.0
        if theta_max is None:
            theta_max = 2 * np.pi * 100.0 / 24000.0
            
        log_theta = torch.linspace(np.log(theta_min), np.log(theta_max), n_channels)
        self.theta = torch.exp(log_theta)
        
        self.radii = torch.linspace(r_min, r_max, n_channels)
        
        # Biquad Coefficients derived from Pole locations
        # z = r * e^(+/- j * theta)
        
        # Denominator: a0=1, a1 = -2r cos(theta), a2 = r^2
        a0 = torch.ones_like(self.radii)
        a1 = -2 * self.radii * torch.cos(self.theta)
        a2 = self.radii ** 2
        
        # Numerator: Gain normalization to roughly 1.0 at peak
        # One simple heuristic: gain ~ 1/(1-r)
        # So we scale by (1-r) to keep it sane.
        b0 = (1 - self.radii) * 0.5 
        
        # Register as buffers (not learnable parameters)
        self.register_buffer('a_coeffs', torch.stack([a0, a1, a2], dim=1).double())
        self.register_buffer('b_coeffs', torch.stack([b0, torch.zeros_like(b0), torch.zeros_like(b0)], dim=1).double())

    def run_filter(self, x):
        # x: [Batch, 1, Time]
        # Expand x for each channel filter
        # x_expanded: [Batch, Channels, Time]
        x_expanded = x.repeat(1, self.a_coeffs.shape[0], 1).double()
        
        # Apply IIR filter
        # torchaudio.functional.lfilter supports broadcasting if shapes align correctly.
        # a_coeffs: [Channels, 3] -> Used across the Time dimension
        return torchaudio.functional.lfilter(x_expanded, self.a_coeffs, self.b_coeffs, clamp=False).float()

    def forward(self, x):
        """
        Processes input x [Batch, 1, Time] through the bank in both forward 
        and backward time directions, concatenating the results.
        Returns: [Batch, Channels * 2, Time]
        """
        fwd_state = self.run_filter(x)
        
        # Flip time for backward pass
        x_flipped = torch.flip(x, dims=[-1])
        
        # Filter and flip back
        bwd_state = torch.flip(self.run_filter(x_flipped), dims=[-1])
        
        return torch.cat([fwd_state, bwd_state], dim=1)

class Tank(nn.Module):
    """
    A random projection mixing layer (Reservoir-like mixing).
    Projects the high-dimensional bi-directional probe state into a 
    dense, non-linear embedding space.
    """
    def __init__(self, in_dim=128, out_dim=128, seed=1337):
        super().__init__()
        torch.manual_seed(seed)
        self.mix = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.mix.weight)

    def forward(self, x):
        # x: [Batch, Channels, Time] -> transpose for Linear -> [Batch, Time, Channels]
        z = self.mix(x.transpose(1, 2))
        # Activation and transpose back
        return torch.tanh(z).transpose(1, 2)


class ESpaceLoss(nn.Module):
    def __init__(self, device='cpu', sr=24000, res_size=64, r_min=0.9999, r_max=0.99999):
        super().__init__()
        
        theta_min = 2 * np.pi * 10.0 / sr
        theta_max = 2 * np.pi * 100.0 / sr

        self.probe = Probe(n_channels=res_size, r_min=r_min, r_max=r_max, theta_min=theta_min, theta_max=theta_max).to(device)
        self.tank = Tank(in_dim=res_size*2, out_dim=res_size).to(device)
        
        # Freeze auxiliary networks
        for p in self.probe.parameters(): p.requires_grad = False
        for p in self.tank.parameters(): p.requires_grad = False
        
    def forward(self, pred, target):
        T = pred.shape[-1]
        margin = T // 3
        
        p_emb = self.tank(self.probe(pred))
        t_emb = self.tank(self.probe(target))
        
        # Slice center to avoid boundary effects
        p_valid = p_emb[..., margin:-margin]
        t_valid = t_emb[..., margin:-margin]
        
        # L1 on State Space
        l_align = F.l1_loss(p_valid, t_valid)
        
        return l_align
