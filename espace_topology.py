import sys
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from loss import MSSLoss
from espace import ESpaceLoss

# -------------------------
# CONFIG
# -------------------------
SR = 24000
VALID_WINDOW = 4096     
MARGIN = 24000          
OFFSET_RANGE = 250
PAD_EXTRA = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    if len(sys.argv) < 3:
        print("Usage: python espace_topology.py <audio_path> <center_idx>")
        sys.exit(1)

    audio_path = sys.argv[1]
    center_idx = int(sys.argv[2])
    
    print(f"Loading {audio_path} on {DEVICE}...")
    data, sr_lib = sf.read(audio_path)
    if data.ndim > 1: data = data[:, 0]
    
    # 1. Prepare Audio Tensor
    # We need enough padding to cover the Margins AND the Search Range
    pad_amt = MARGIN + VALID_WINDOW + OFFSET_RANGE + PAD_EXTRA
    data = np.pad(data, (pad_amt, pad_amt), mode='reflect')
    shifted_center = center_idx + pad_amt
    
    full_audio = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE) # [1, 1, T]

    # 2. Setup Loss
    criterion = ESpaceLoss(device=DEVICE, sr=SR).to(DEVICE)
    loss_mss = MSSLoss().to(DEVICE)
    
    # 3. Pre-compute TARGET Embedding (Anchor)
    print("Computing Target Embedding...")
    with torch.no_grad():
        # Indices for the Target Window (Context + Valid + Context)
        start_t = shifted_center - (VALID_WINDOW // 2) - MARGIN
        end_t   = start_t + MARGIN + VALID_WINDOW + MARGIN
        
        target_chunk = full_audio[..., start_t : end_t]
        
        # Run Probe ONCE for target
        p_t = criterion.probe(target_chunk)
        t_t = criterion.tank(p_t)
        
        # Crop to valid region (center)
        target_emb_valid = t_t[..., MARGIN : -MARGIN]

    # 4. Pre-compute SEARCH SPACE Embedding
    # We define a "Super Chunk" that covers the entire sweep range.
    print("Pre-computing Search Space...")
    with torch.no_grad():
        sweep_start_idx = start_t - OFFSET_RANGE
        sweep_end_idx   = end_t + OFFSET_RANGE
        
        super_chunk = full_audio[..., sweep_start_idx : sweep_end_idx]
        
        # Run IIR ONCE over the massive chunk
        # This settles the filters for the entire neighborhood
        p_super = criterion.probe(super_chunk)
        t_super = criterion.tank(p_super)
        
        # t_super is now a map of embeddings for the whole search area.
        # Shape: [1, Channels, (TotalLen + 2*OffsetRange)]

    # 5. Fast Loop (Slicing)
    offsets = range(-OFFSET_RANGE, OFFSET_RANGE + 1)
    
    l_align_hist = []
    l_mss_hist = [] 
    l1_hist = []
    
    print(f"Sweeping {len(offsets)} offsets using GPU slicing...")
    
    with torch.no_grad():
        for off in offsets:
            # --- A. Alignment Loss (O(1) Slice) ---
            # We need to find where the "valid window" for this offset lives inside t_super.
            # super_chunk starts at (start_t - OFFSET_RANGE)
            # current window starts at (start_t + off)
            
            # The "Valid" region (post-margin) is offset by MARGIN inside the window.
            idx_start = (off + OFFSET_RANGE) + MARGIN
            idx_end   = idx_start + VALID_WINDOW
            
            curr_emb_valid = t_super[..., idx_start : idx_end]
            
            # Cosine Loss
            val_align = 1.0 - F.cosine_similarity(curr_emb_valid, target_emb_valid, dim=1).mean().item()
            l_align_hist.append(val_align)
            
            # --- B. MSS / L1 (Must extract raw audio) ---
            # Current raw chunk start (absolute)
            c_s = start_t + off
            c_e = end_t + off
            
            # We only need the center valid part for L1/MSS comparison
            raw_curr = full_audio[..., c_s + MARGIN : c_e - MARGIN]
            raw_targ = target_chunk[..., MARGIN : -MARGIN]
            
            l1_val = F.l1_loss(raw_curr, raw_targ).item()
            l1_hist.append(l1_val)
            
            mss_val = loss_mss(raw_curr, raw_targ).item()
            l_mss_hist.append(mss_val)

    # 6. Plotting
    def norm(x):
        a = np.array(x)
        return (a - a.min()) / (a.max() - a.min() + 1e-8)

    plt.figure(figsize=(12, 8))
    plt.plot(offsets, norm(l1_hist), label="L1 (Raw)", color="red", alpha=0.3, linestyle="--")
    plt.plot(offsets, norm(l_mss_hist), label="MSS", color="green", alpha=0.4, linestyle="--")
    plt.plot(offsets, norm(l_align_hist), label="ESpace Alignment", color="blue", linewidth=3)
    
    plt.title(f"Fast ESpace Topology (GPU)\nCenter: {center_idx}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/espace_topology.png")
    print("Done. Saved results/espace_topology.png")

if __name__ == "__main__":
    main()