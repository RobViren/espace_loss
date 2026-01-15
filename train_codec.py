import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import numpy as np
from espace_loss import ESpaceLoss

from pathlib import Path

# ==========================================
# CONSTANTS
# ==========================================
SR = 24000
BATCH_SIZE = 8
LR = 8e-4
MAX_STEPS = 50000
CHECKPOINT_INTERVAL = 250
LOG_INTERVAL = 20
SEGMENT_SIZE = 72000
ALIGN_WEIGHT = 10.0
RES_SIZE = 64

# ==========================================
# Codec Architecture
# ==========================================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(),
            CausalConv1d(channels, channels, 3, dilation=dilation),
            nn.ELU(),
            CausalConv1d(channels, channels, 1)
        )
    def forward(self, x):
        return x + self.block(x)

class UpSampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return self.conv(x)

class SimpleCodec(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        
        # Encoder (Downsample 8x)
        self.encoder = nn.Sequential(
            CausalConv1d(1, channels, 7, stride=1),
            ResBlock(channels),
            CausalConv1d(channels, channels*2, 4, stride=2), 
            ResBlock(channels*2),
            CausalConv1d(channels*2, channels*4, 4, stride=2), 
            ResBlock(channels*4),
            CausalConv1d(channels*4, channels*8, 4, stride=2), 
        )
        
        self.bottleneck = nn.Conv1d(channels*8, channels*8, 3, padding=1)
        
        # Decoder (Upsample 8x)
        self.decoder = nn.Sequential(
            UpSampleBlock(channels*8, channels*4, 2),
            ResBlock(channels*4),
            UpSampleBlock(channels*4, channels*2, 2),
            ResBlock(channels*2),
            UpSampleBlock(channels*2, channels, 2),
            ResBlock(channels),
            nn.Conv1d(channels, 1, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(self.bottleneck(z))
        if out.shape[-1] != x.shape[-1]:
            out = out[..., :x.shape[-1]]
        return out

# ==========================================
# Helpers
# ==========================================
def si_snr(estimate, reference):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    alpha = (estimate * reference).sum() / (reference * reference).sum()
    target = alpha * reference
    noise = estimate - target
    return 10 * torch.log10((target ** 2).sum() / ((noise ** 2).sum() + 1e-9))

class LibriStream(IterableDataset):
    def __init__(self, split="train.clean.100", segment_size=SEGMENT_SIZE):
        self.ds = load_dataset("mythicinfinity/libritts", "clean", split=split, streaming=True)
        self.segment_size = segment_size
        self.sr = SR
    def __iter__(self):
        for item in self.ds:
            try:
                audio = torch.from_numpy(item['audio']['array']).float()
                if item['audio']['sampling_rate'] != self.sr:
                    audio = torchaudio.transforms.Resample(item['audio']['sampling_rate'], self.sr)(audio)
                if audio.shape[0] > self.segment_size:
                    for _ in range(2):
                        start = torch.randint(0, audio.shape[0]-self.segment_size, (1,)).item()
                        yield (audio[start:start+self.segment_size] / (audio.abs().max()+1e-6)).unsqueeze(0)
            except: continue

def load_validation_sample(path, device):
    print(f"Loading validation reference: {path}")
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
    if sr != SR: audio = torchaudio.transforms.Resample(sr, SR)(audio)
    if audio.shape[-1] < SEGMENT_SIZE: audio = F.pad(audio, (0, SEGMENT_SIZE - audio.shape[-1]))
    return (audio[..., :SEGMENT_SIZE] / audio.abs().max()).to(device).unsqueeze(0)

# ==========================================
# Training Loop
# ==========================================
def train(validation_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Codec on {device} ---")
    
    ds = LibriStream()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
    
    model = SimpleCodec().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    criterion = ESpaceLoss(device=device, sr=SR, align_weight=ALIGN_WEIGHT).to(device)
    
    val_audio = load_validation_sample(validation_path, device)
    os.makedirs("results/training", exist_ok=True)
    
    step = 0
    iterator = iter(loader)
    
    print("Starting Training (L1 Holographic Mode)...")
    
    while step < MAX_STEPS:
        try:
            x = next(iterator).to(device)
        except StopIteration:
            iterator = iter(loader)
            x = next(iterator).to(device)
            
        x_recon = model(x)
        loss, l_mss, l_align = criterion(x_recon, x)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % LOG_INTERVAL == 0:
            print(f"Step {step:05d} | Total: {loss.item():.3f} | Spec: {l_mss:.3f} | Align: {l_align:.3f}")
            
        if step % CHECKPOINT_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                val_recon = model(val_audio)
                snr = si_snr(val_recon.squeeze(), val_audio.squeeze())
                path = f"results/training/step_{step}.wav"
                torchaudio.save(path, val_recon.squeeze().cpu(), SR)
            print(f"--> Saved {path} | Val SI-SNR: {snr.item():.2f} dB")
            model.train()
            
        step += 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_codec.py <validation_wav_path>")
        sys.exit(1)
    directory = Path("results/training")
    directory.mkdir(parents=True, exist_ok=True)
    train(sys.argv[1])