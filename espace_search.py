import sys
import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from espace_loss import ESpaceLoss

from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
SAMPLE_RATE = 24000
WINDOW_SAMPLES = 4096
RESULTS_DIR = "results"

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_audio_torch(path):
    print(f"Loading {path}...")
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    max_val = np.max(np.abs(audio))
    if max_val > 0: audio = audio / max_val
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0) # (1, Time)

def plot_search_analysis(q_audio, t_audio, dist_total, dist_fwd, dist_bwd, q_idx, m_idx, fname):
    w = WINDOW_SAMPLES
    hw = w // 2
    
    # 1. Slice Waveforms for Overlay
    # Handle boundary conditions
    s_q, e_q = max(0, q_idx-hw), min(q_audio.shape[-1], q_idx+hw)
    s_m, e_m = max(0, m_idx-hw), min(t_audio.shape[-1], m_idx+hw)
    
    q_chunk = q_audio.squeeze().numpy()[s_q:e_q]
    m_chunk = t_audio.squeeze().numpy()[s_m:e_m]
    
    # 2. Slice Distances (Centered on Match)
    d_total = dist_total[s_m:e_m]
    d_fwd = dist_fwd[s_m:e_m]
    d_bwd = dist_bwd[s_m:e_m]
    
    # Create time axis relative to match
    t_axis = np.arange(len(m_chunk)) - (m_idx - s_m) 

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
    
    # --- Plot 1: Waveform Alignment ---
    ax1.plot(t_axis[:len(q_chunk)], q_chunk, 'k', linewidth=1.5, label='Query')
    ax1.plot(t_axis, m_chunk, 'r--', linewidth=1.5, label='Match')
    ax1.set_title(f"Waveform Alignment (Offset: {m_idx - q_idx})")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Loss Landscape ---
    ax2.plot(t_axis, d_fwd, color='orange', linestyle=':', label='Channel Group A')
    ax2.plot(t_axis, d_bwd, color='green', linestyle=':', label='Channel Group B')
    ax2.plot(t_axis, d_total, color='blue', linewidth=2, label='Total E-Space Distance')
    
    ax2.set_title("E-Space Loss Landscape (Phase Locking Basin)")
    ax2.set_ylabel("Euclidean Distance")
    ax2.set_xlabel("Offset from Match (Samples)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mark the bottom of the bowl
    min_val = d_total[np.argmin(d_total)]
    ax2.scatter([0], [min_val], c='red', zorder=10)
    
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Saved visualization to {fname}")

def run(q_path, t_path, q_idx):
    # Load
    q_audio = load_audio_torch(q_path)
    t_audio = load_audio_torch(t_path)
    
    # Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # Use the loss components to generate embeddings
    loss_module = ESpaceLoss(device=device, sr=SAMPLE_RATE).to(device)
    
    print("Processing E-Space...")
    with torch.no_grad():
        # Forward Pass via Probe -> Tank
        # Shapes: (1, Channels, Time)
        q_states = loss_module.tank(loss_module.probe(q_audio.to(device).unsqueeze(0)))
        t_states = loss_module.tank(loss_module.probe(t_audio.to(device).unsqueeze(0)))
        
        # Extract Query Vector at Index
        # Shape: (Channels)
        q_vec = q_states[0, :, q_idx]
        
        # --- Calculate Distances ---
        # Target: (1, Channels, Time)
        # Query:  (Channels) -> broadcast to (1, Channels, Time)
        
        diff = t_states - q_vec.unsqueeze(0).unsqueeze(2)
        
        # 1. Total Distance (L2 Norm of all channels)
        dist_total = torch.linalg.vector_norm(diff, ord=2, dim=1).squeeze().cpu().numpy()
        
        # 2. Split Distances for Visualization (Arbitrary channel split)
        total_channels = diff.shape[1]
        mid = total_channels // 2
        diff_fwd = diff[:, :mid, :]
        diff_bwd = diff[:, mid:, :]
        
        dist_fwd = torch.linalg.vector_norm(diff_fwd, ord=2, dim=1).squeeze().cpu().numpy()
        dist_bwd = torch.linalg.vector_norm(diff_bwd, ord=2, dim=1).squeeze().cpu().numpy()
        
        # --- Find Best Match ---
        match_idx = np.argmin(dist_total)
        print(f"Query Sample: {q_idx}")
        print(f"Best Match:   {match_idx}")
        print(f"Distance:     {dist_total[match_idx]:.6f}")
        
        ensure_dir(RESULTS_DIR)
        out_name = os.path.join(RESULTS_DIR, f"espace_search_q{q_idx}.png")
        plot_search_analysis(q_audio, t_audio, dist_total, dist_fwd, dist_bwd, q_idx, match_idx, out_name)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python espace_search.py <query_wav> <target_wav> <query_idx>")
        sys.exit(1)
    directory = Path("results")
    directory.mkdir(parents=True, exist_ok=True)
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))