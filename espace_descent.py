import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from espace import ESpaceLoss
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
SR = 24000
VALID_WINDOW = 4096
MARGIN = 24000 
START_OFFSET = 300.0 
STEPS = 200     
LR = 5.0
SEARCH_PAD = 1000
SWEEP_RANGE = 350

def differentiable_resample(data, start_idx, window_size):
    """
    Extracts a window from a multi-channel tensor 'data' [1, C, T]
    at float index 'start_idx' using linear interpolation.
    """
    B, C, T = data.shape
    device = data.device
    
    # Grid of offsets [0, 1, ... W-1]
    grid = torch.arange(window_size, device=device).float()
    
    # Target indices [Batch, Window]
    idx = start_idx.reshape(1, 1) + grid.unsqueeze(0)
    
    idx_floor = torch.floor(idx).long()
    idx_ceil = idx_floor + 1
    
    idx_floor = torch.clamp(idx_floor, 0, T - 1)
    idx_ceil = torch.clamp(idx_ceil, 0, T - 1)
    
    alpha = idx - idx_floor.float() # [1, Window]
    
    # Explicit Slicing for B=1 to avoid 4D tensor hell
    val_floor = data[0][:, idx_floor[0]]
    val_ceil  = data[0][:, idx_ceil[0]]
    
    # Interpolate (Alpha broadcasts correctly now: [1, W] * [C, W])
    output = (1 - alpha) * val_floor + alpha * val_ceil
    
    return output.unsqueeze(0) # [1, C, Window]

def main():
    if len(sys.argv) < 3:
        print("Usage: python espace_descent.py <audio_path> <center_idx>")
        sys.exit(1)

    audio_path = sys.argv[1]
    center_idx = int(sys.argv[2])
    
    print("Loading audio...")
    data, sr = sf.read(audio_path)
    if data.ndim > 1: data = data[:, 0]
    
    # Pad aggressively
    pad_amt = MARGIN + SEARCH_PAD
    data = np.pad(data, (pad_amt, pad_amt), mode='reflect')
    
    # Shift center to match padded data
    center_idx += pad_amt
    
    audio_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 1. Setup Loss (Align only)
    criterion = ESpaceLoss(sr=SR).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # 2. Pre-compute Smooth Embeddings
    print("Pre-computing smooth embeddings...")
    with torch.no_grad():
        p_full = criterion.probe(audio_tensor)
        t_full = criterion.tank(p_full)
    
    # 3. Setup Target Embedding
    # Target starts at: center - half_valid
    target_start_idx = center_idx - (VALID_WINDOW // 2)
    
    with torch.no_grad():
        t_idx = int(target_start_idx)
        # Slicing strictly ensures shape [1, C, Window]
        target_emb = t_full[..., t_idx : t_idx + VALID_WINDOW]

    # 4. Optimizer
    offset_param = nn.Parameter(torch.tensor([float(START_OFFSET)]))
    optimizer = optim.Adam([offset_param], lr=LR)
    
    print(f"--- STARTING SMOOTH DESCENT ---")
    
    history = []
    loss_history = []
    
    for i in range(STEPS):
        optimizer.zero_grad()
        
        # Current read head
        curr_read_idx = float(target_start_idx) + offset_param
        
        # Resample
        curr_emb = differentiable_resample(t_full, curr_read_idx, VALID_WINDOW)
        
        # Cosine Loss
        loss = 1.0 - F.cosine_similarity(curr_emb, target_emb, dim=1).mean()
        
        loss.backward()
        optimizer.step()
        
        history.append(offset_param.item())
        loss_history.append(loss.item())
        
        if i % 20 == 0:
            print(f"Step {i:03d} | Offset: {offset_param.item():6.2f} | Loss: {loss.item():.7f}")

    print(f"Final Offset: {history[-1]:.4f}")
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    
    sweep = np.arange(-SWEEP_RANGE, SWEEP_RANGE, 1)
    landscape = []
    print("Scanning landscape...")
    with torch.no_grad():
        for s in sweep:
            ridx = torch.tensor([float(target_start_idx + s)])
            e = differentiable_resample(t_full, ridx, VALID_WINDOW)
            l = 1.0 - F.cosine_similarity(e, target_emb, dim=1).mean()
            landscape.append(l.item())
        
    plt.plot(sweep, landscape, color='gray', alpha=0.5, label='Basin')
    plt.plot(history, loss_history, 'o-', color='red', label='Descent', markersize=3)
    plt.plot(history[0], loss_history[0], 'go', label='Start')
    plt.plot(history[-1], loss_history[-1], 'bo', label='End')
    
    plt.title(f"Descent on Smooth Embeddings\nFinal: {history[-1]:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/descent.png")
    print("Saved results/descent.png")

if __name__ == "__main__":
    directory = Path("results")
    directory.mkdir(parents=True, exist_ok=True)
    main()