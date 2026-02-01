import torch
import torch.nn as nn
import math

class RateDistortionLoss(nn.Module):
    """
    Standard Loss function for Neural Video Compression.
    L = lambda * Distortion + Rate
    
    This aligns with the logic found in the training notebooks:
    loss = lambda * mse_loss + bpp_loss
    """
    def __init__(self, lambda_val=256):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_val = lambda_val

    def forward(self, output, target):
        """
        Calculates the joint RD Loss.
        
        Args:
            output (dict): Contains "x_hat" (reconstruction) and "likelihoods".
            target (Tensor): The ground truth image.
            
        Returns:
            total_loss, distortion, bpp
        """
        # 1. Distortion (MSE)
        x_hat = output["x_hat"]
        distortion = self.mse_loss(x_hat, target)
        
        # 2. Rate (BPP - Bits Per Pixel)
        # Sum of log-likelihoods from Motion (y,z) and Residual (y,z) latents
        likelihoods = output["likelihoods"]
        bpp_loss = 0
        num_pixels = x_hat.numel()
        
        # Standard CompressAI BPP calculation
        for v in likelihoods.values():
            # Entropy calculation: -log2(p)
            bpp_loss += torch.log(v).sum() / (-math.log(2) * num_pixels)
            
        # Total Loss formulation
        loss = self.lambda_val * distortion + bpp_loss
        
        return loss, distortion, bpp_loss