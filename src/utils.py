import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_adaptive_mask(residual, lambda_param=1.5, epsilon=1e-6):
    """
    Computes the spatial fusion mask M(p) based on the energy of the residual.
    Used by AdaptiveRefiNET to decide how to blend prediction and residual.
    
    Args:
        residual (Tensor): The residual feature map [B, C, H, W].
        lambda_param (float): Sensitivity factor.
        epsilon (float): Small constant to avoid division by zero.
        
    Returns:
        mask (Tensor): The computed soft mask [B, 1, H, W].
    """
    # Calculate L2 norm per pixel across channels
    norm_per_pixel = torch.sqrt(torch.sum(residual ** 2, dim=1, keepdim=True))
    
    H, W = residual.shape[2], residual.shape[3]
    
    # Calculate global mean of the norm
    mu = torch.sum(norm_per_pixel, dim=(2, 3), keepdim=True) / (H * W)
    
    # Apply hyperbolic tangent to get values in range [0, 1] (soft mask)
    mask = torch.tanh(lambda_param * norm_per_pixel / (mu + epsilon))
    
    return mask

def flow_warp(x, flow):
    """
    Differentiable warping operation using Optical Flow.
    Moves pixels from 'x' according to vectors in 'flow'.
    
    Args:
        x (Tensor): Input image or feature map [B, C, H, W].
        flow (Tensor): Optical Flow field [B, 2, H, W].
        
    Returns:
        Tensor: Warped image.
    """
    B, C, H, W = x.size()
    
    # Generate a standard meshgrid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    grid = torch.cat((xx.view(1, 1, H, W).repeat(B, 1, 1, 1), 
                      yy.view(1, 1, H, W).repeat(B, 1, 1, 1)), 1).float().to(x.device)
    
    # Apply flow to the grid
    vgrid = grid + flow
    
    # Normalize coordinates to range [-1, 1] for grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    # Bilinear sampling
    return F.grid_sample(x, vgrid.permute(0, 2, 3, 1), mode='bilinear', align_corners=True)

def robust_load(model, path):
    """
    Loads model weights safely, handling 'module.' prefixes from DataParallel training
    and ignoring mismatching keys (strict=False).
    """
    if not path:
        print("No path provided for weights loading.")
        return

    print(f"Loading weights from {path}...")
    # Load on CPU first to avoid CUDA OOM during initialization
    checkpoint = torch.load(path, map_location='cpu')
    
    # Handle dictionary structure
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights
    model.load_state_dict(new_state_dict, strict=False)
    print("Weights loaded successfully.")