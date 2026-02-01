"""
Main training script for the Neural Video Compression framework.

This script performs the Joint Fine-Tuning (Stage 3) of the entire pipeline:
- Motion VAE (Pre-trained on Vimeo-Triplet/HardMode) -> Unfrozen
- Residual VAE (Pre-trained on Vimeo-Septuplet) -> Unfrozen
- Adaptive Fusion Module -> Learned from scratch

It requires pre-trained weights in the 'weights/' directory to initialize the backbone.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import local modules from src/
from src.models import ScaleHyperprior, MotionRefineNET, ResRefiNET, AdaptiveRefiNET
from src.dataset import VimeoSeptupletDataset, VimeoTripletSkipDataset, VimeoHardModeDataset
from src.loss import RateDistortionLoss
from src.utils import robust_load, compute_adaptive_mask, flow_warp
from configs.config import Config

# Try to import RAFT (optional for showcase, but good to have)
try:
    from torchvision.models.optical_flow import raft_small
    RAFT_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False
    print("  RAFT not found. Using dummy flow for demonstration.")

def train_joint():
    # 1. SETUP DEVICE & DIRS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Starting Joint Fine-Tuning on {device}")
    
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # 2. INITIALIZE ARCHITECTURE (The "Pentagon")
    print("  Building Model Architecture...")
    
    # --- MOTION BRANCH (Larger Capacity: N=192) ---
    # Based on pre-trained weights 'FlowVAE_finetune_ep11.pth'
    print(" Init Motion Model (N=192, M=192)")
    motion_model = ScaleHyperprior(N=192, M=192, in_channels=2, out_channels=2).to(device)
    
    refine_model = MotionRefineNET().to(device)
    
    # --- RESIDUAL BRANCH (Standard Capacity: N=128) ---
    # Based on pre-trained weights 'ResidualVAE_HardMode_Ep4.pth'
    print("🔹 Init Residual Model (N=128, M=128)")
    residual_model = ScaleHyperprior(N=128, M=128, in_channels=3, out_channels=3).to(device)
    
    # --- RECONSTRUCTION BRANCH ---
    adaptive_model = AdaptiveRefiNET().to(device)
    post_model = ResRefiNET().to(device)
    
    # Optical Flow Backbone (RAFT Small - Frozen)
    if RAFT_AVAILABLE:
        raft = raft_small(pretrained=True).to(device).eval()
        for p in raft.parameters(): p.requires_grad = False

    # 3. LOAD PRE-TRAINED WEIGHTS (Stage-wise Initialization)
    print(" Loading Pre-trained Weights...")
    robust_load(motion_model,   Config.PRETRAINED_WEIGHTS['motion'])
    robust_load(refine_model,   Config.PRETRAINED_WEIGHTS['refine'])
    robust_load(residual_model, Config.PRETRAINED_WEIGHTS['residual'])
    
    # Adaptive & Post might be new or pre-trained
    if os.path.exists(Config.PRETRAINED_WEIGHTS['adaptive']):
        robust_load(adaptive_model, Config.PRETRAINED_WEIGHTS['adaptive'])
    else:
        print(" Adaptive weights not found (starting from scratch)")

    if os.path.exists(Config.PRETRAINED_WEIGHTS['post']):
        robust_load(post_model, Config.PRETRAINED_WEIGHTS['post'])
    else:
        print(" Post-process weights not found (starting from scratch)")

    # 4. OPTIMIZER SETUP (Split Net vs Aux)
    # Critical: CompressAI models have "auxiliary" parameters (quantiles) 
    # that must be optimized separately to estimate entropy correctly.
    
    net_params = []
    aux_params = []
    
    # Collect parameters from all trainable modules
    model_list = [motion_model, refine_model, residual_model, adaptive_model, post_model]
    
    for model in model_list:
        for n, p in model.named_parameters():
            if p.requires_grad:
                if n.endswith(".quantiles"):
                    aux_params.append(p)
                else:
                    net_params.append(p)

    # Main Optimizer (Weights)
    optimizer = optim.AdamW(net_params, lr=Config.LEARNING_RATE)
    
    # Aux Optimizer (Entropy Bottleneck parameters) - usually requires higher LR (1e-3)
    aux_optimizer = optim.Adam(aux_params, lr=1e-3)
    
    criterion = RateDistortionLoss(lambda_val=Config.LAMBDA_RD)

    # 5. DATASET SELECTION
    print(f" Loading Dataset... [Mode: {Config.DATASET_MODE}]")
    
    if Config.DATASET_MODE == "septuplet":
        dataset = VimeoSeptupletDataset(Config.DATASET_DIR, split='train', crop_size=Config.CROP_SIZE)
    elif Config.DATASET_MODE == "triplet":
        dataset = VimeoTripletSkipDataset(Config.DATASET_DIR, split='train', crop_size=Config.CROP_SIZE)
    elif Config.DATASET_MODE == "hard_mode":
        dataset = VimeoHardModeDataset(Config.DATASET_DIR, gap=Config.HARD_MODE_GAP, split='train', crop_size=Config.CROP_SIZE)
    else:
        raise ValueError(f"Unknown dataset mode: {Config.DATASET_MODE}")

    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)

    # 6. TRAINING LOOP
    print(f" Starting Training Loop for {Config.EPOCHS} Epochs...")
    
    # Set models to train mode
    for model in model_list:
        model.train()

    for epoch in range(Config.EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for batch_idx, (frame1, frame2) in enumerate(loop):
            frame1, frame2 = frame1.to(device), frame2.to(device)
            
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            
            # --- A. OPTICAL FLOW ESTIMATION ---
            if RAFT_AVAILABLE:
                with torch.no_grad():
                    # RAFT estimation (im1 -> im2)
                    flow_pred = raft(frame1, frame2)[-1] 
            else:
                # Dummy flow
                flow_pred = torch.zeros(frame1.shape[0], 2, frame1.shape[2], frame1.shape[3]).to(device)

            # --- B. MOTION COMPRESSION ---
            motion_out = motion_model(flow_pred)
            flow_hat = motion_out["x_hat"]
            motion_likelihoods = motion_out["likelihoods"]
            
            # --- C. WARPING & REFINEMENT ---
            frame2_pred = flow_warp(frame1, flow_hat)
            
            # Zero-history for pair-based training
            history_dummy = torch.zeros(frame1.shape[0], 12, frame1.shape[2], frame1.shape[3]).to(device)
            frame2_refined = refine_model(frame2_pred, history_dummy)
            
            # --- D. RESIDUAL CODING ---
            residual = frame2 - frame2_refined
            res_out = residual_model(residual)
            res_hat = res_out["x_hat"]
            res_likelihoods = res_out["likelihoods"]
            
            recon_residual = res_hat 
            
            # --- E. ADAPTIVE FUSION ---
            mask = compute_adaptive_mask(recon_residual)
            frame_recon = adaptive_model(recon_residual, frame2_refined, mask, history_dummy)
            
            # --- F. POST-PROCESSING ---
            final_image = post_model(frame_recon)
            
            # --- G. MAIN LOSS & BACKPROP ---
            combined_likelihoods = {**motion_likelihoods, **res_likelihoods}
            output_dict = {
                "x_hat": final_image,
                "likelihoods": combined_likelihoods
            }
            
            loss, dist, bpp = criterion(output_dict, frame2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_params, 1.0)
            optimizer.step()
            
            # --- H. AUXILIARY LOSS OPTIMIZATION (Crucial Step) ---
            # Update the entropy bottleneck parameters (CDF tables)
            aux_loss = motion_model.aux_loss() + residual_model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            
            # Update Progress Bar
            loop.set_postfix(loss=loss.item(), bpp=bpp.item(), dist=dist.item(), aux=aux_loss.item())

        # Save Checkpoint
        save_path = os.path.join(Config.CHECKPOINT_DIR, f"joint_model_ep{epoch+1}.pth")
        
        torch.save({
            'epoch': epoch,
            'motion_state': motion_model.state_dict(),
            'residual_state': residual_model.state_dict(),
            'adaptive_state': adaptive_model.state_dict(),
            'post_state': post_model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_path)
        
        print(f" Checkpoint saved: {save_path}")

if __name__ == "__main__":
    train_joint()