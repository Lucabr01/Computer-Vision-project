import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN

# ============================================================================
# 1. HELPER LAYERS (Basic Blocks)
# ============================================================================

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    """Standard strided convolution for downsampling."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    """Transposed convolution for upsampling."""
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class ResBlock(nn.Module):
    """
    Standard Residual Block with optional Group Normalization.
    Used in MotionRefineNET and ResRefiNET.
    """
    def __init__(self, c, dilation=1, use_gn=True):
        super().__init__()
        self.use_gn = use_gn
        self.conv1 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        
        if use_gn:
            self.gn1 = nn.GroupNorm(8, c)
            self.gn2 = nn.GroupNorm(8, c)
            
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        if self.use_gn: y = self.gn1(y)
        y = self.act(y)
        y = self.conv2(y)
        if self.use_gn: y = self.gn2(y)
        return x + y

class SimpleResBlock(nn.Module):
    """
    Lightweight Residual Block (no Norm) for AdaptiveRefiNET.
    Optimized for speed and memory.
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        
    def forward(self, x):
        return x + self.block(x)


# ============================================================================
# 2. VAE ARCHITECTURE (Motion & Residual Compressor)
# ============================================================================

class ScaleHyperprior(CompressionModel):
    """
    Variational Autoencoder (VAE) with Scale Hyperprior mechanism.
    Used for:
    1. Motion Compression (in_channels=2)
    2. Residual Compression (in_channels=3)
    """
    def __init__(self, N, M, in_channels=3, out_channels=3, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

        # Main Encoder (g_a)
        self.g_a = nn.Sequential(
            conv(in_channels, N), GDN(N),
            conv(N, N), GDN(N),
            conv(N, N), GDN(N),
            conv(N, M),
        )

        # Main Decoder (g_s)
        self.g_s = nn.Sequential(
            deconv(M, N), GDN(N, inverse=True),
            deconv(N, N), GDN(N, inverse=True),
            deconv(N, N), GDN(N, inverse=True),
            deconv(N, out_channels),
        )

        # Hyperprior Encoder (h_a)
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3), nn.ReLU(inplace=True),
            conv(N, N), nn.ReLU(inplace=True),
            conv(N, N),
        )
        
        # Hyperprior Decoder (h_s)
        self.h_s = nn.Sequential(
            deconv(N, N), nn.ReLU(inplace=True),
            deconv(N, N), nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3), nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    def forward(self, x):
        """Forward pass: compresses and decompresses 'x'."""
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        
        # Compress latents
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        
        # Reconstruct
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Factory method to load model structure from weights file."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        in_channels = state_dict["g_a.0.weight"].size(1) 
        out_channels = state_dict["g_s.6.weight"].size(0)
        
        net = cls(N, M, in_channels=in_channels, out_channels=out_channels)
        net.load_state_dict(state_dict)
        return net


# ============================================================================
# 3. REFINEMENT MODULES
# ============================================================================

class MotionRefineNET(nn.Module):
    """
    Refines the decoded optical flow using temporal history context.
    Weights file: RefiFlow.pth
    """
    def __init__(self, base=64, blocks=8, use_gn=True, use_gate=True):
        super().__init__()
        # Input channels: 2 (Flow) + 12 (History 4 frames) = 14 Channels
        in_ch = 14 
        
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True))
        
        body = []
        for i in range(blocks):
            dil = 1 if i < blocks - 2 else 2
            body.append(ResBlock(base, dilation=dil, use_gn=use_gn))
        self.body = nn.Sequential(*body)

        self.delta_head = nn.Conv2d(base, 2, 3, padding=1)
        nn.init.zeros_(self.delta_head.weight) # Start learning from zero delta
        nn.init.zeros_(self.delta_head.bias)

        self.use_gate = use_gate
        if use_gate:
            self.gate_head = nn.Conv2d(base, 1, 3, padding=1)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)

    def forward(self, flow_hat, history_4f):
        # Concatenate current flow with 4 previous frames
        x = torch.cat([flow_hat, history_4f], dim=1)
        
        f = self.stem(x)
        f = self.body(f)
        
        delta = self.delta_head(f)
        
        # Apply gating mechanism if enabled
        if self.use_gate:
            gate = torch.sigmoid(self.gate_head(f))
            delta = gate * delta
            
        return flow_hat + delta


class ResRefiNET(nn.Module):
    """
    Post-processing network to remove blocking artifacts from the final image.
    Weights file: ResRefiNET.pth
    """
    def __init__(self, in_channels=3, mid_channels=64, num_blocks=6):
        super().__init__()
        self.head = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResBlock(mid_channels, use_gn=True) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1)
        
        # Initialize tail to zero to start as an identity mapping
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        identity = x
        out = self.head(x)
        out = self.body(out)
        correction = self.tail(out)
        return identity + correction


class AdaptiveRefiNET(nn.Module):   
    """
    Advanced Fusion Module (Adaptive Mask).
    Merges the Residual reconstruction and the Warped frame based on a dynamic mask.
    Weights file: AdaptiveNET.pth
    """
    def __init__(self, base=64, num_blocks=10):
        super().__init__()
        
        # Input: 19 Channels 
        # 3 (Recon) + 3 (Warped) + 1 (Mask) + 12 (History 4 frames) = 19
        self.encoder = nn.Sequential(
            nn.Conv2d(19, base, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Lightweight residual blocks
        self.res_blocks = nn.ModuleList([
            SimpleResBlock(base) for _ in range(num_blocks)
        ])
        
        self.decoder = nn.Sequential(
            nn.Conv2d(base, base, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, 3, 3, 1, 1)
        )
        
    def forward(self, recon, warped, mask, history):
        # Concatenate all inputs along the channel dimension
        x = torch.cat([recon, warped, mask, history], dim=1)
        
        feat = self.encoder(x)
        
        for block in self.res_blocks:
            feat = block(feat)
        
        correction = self.decoder(feat)
        
        # Apply correction weighted by the mask
        # If mask is high -> trust the correction more
        # If mask is low  -> trust the original recon more
        refined = recon + correction * mask
        
        return refined.clamp(0, 1)