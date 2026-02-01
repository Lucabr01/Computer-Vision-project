# Optical Flow-based Neural Video Compression

1. [Introduction](#1-introduction)
   1. [Fundamentals](#11-fundamentals)
2. [Codec Architecture](#2-codec-architecture)
   1. [Proposed Architecture](#21-proposed-architecture)

# 1. Introduction

Our project aims to build a *Low-bitrate Lossy Video Codec* that replicates and enhances the conventional video compression architecture. Traditional manually designed codecs rely on motion vectors and residual information to reconstruct non-key frames from previously decoded frames. The reason is that, as will be discussed later, these two pieces of information account for only a small fraction of the original frame size, yet they are sufficient to reconstruct it with high fidelity.

The paper implemented in this project is **[DVC: An End-to-End Deep Video Compression Framework](https://arxiv.org/pdf/1812.00101)** [1]. It represents one of the first neural-network-based approaches to video coding and introduces the general architecture adopted by many subsequent methods. Our goal was to understand this paradigm and extend it by addressing some of the limitations of the original DVC architecture, such as occlusions and the lack of explicit temporal context.

## 1.1 Fundamentals

Lets see all the main logical components for our codec:

### Optical Flow

Given two consecutive frames, the **Optical Flow** quantifies the motion of objects between them. In simple terms, optical flow describes the direction and magnitude (the speed) of the displacement of each pixel from the first frame to the second. An example: 

<p align="center">
  <img src="images/Football.png" alt="Training Curves" width="80%">
</p>

The left image shows motion vectors, where the direction and length of each arrow represent the direction and magnitude of the players displacement. The right image shows the corresponding flow magnitude map, highlighting regions with significant motion while suppressing static background areas.

### Warping operation

Using the optical flow vectors, the first image is warped to match the second image. So the **Warping Operation** consists in: shifting each pixel in the first image according to the direction and magnitude given by the flow, effectively predicting where that pixel will be in the next frame. An example:

Aggiungere foto

### Residual

As can be seen from the previous example, warping an image based solely on optical flow often introduces artifacts such as occlusions, ghosting effects, and inaccurate motion estimation in complex regions. For this reason, a residual signal is computed to capture the information that cannot be explained by motion compensation alone. In simple terms, the residual error is computed as the difference between the original frame and its warped prediction. Example: 

Aggiungere foto

### PSNR (Peak Signal-to-Noise Ratio)

In image and video compression, reconstruction quality is commonly evaluated using the **Mean Squared Error (MSE)**, which measures the average squared difference between the original and reconstructed images. While MSE is simple and directly optimized during training, its values are often difficult to interpret in terms of perceptual quality.

For this reason, the **PSNR** is widely used. PSNR is a logarithmic transformation of the MSE that expresses reconstruction quality in **decibels (dB)**:

PSNR = 10 · log10(MAX² / MSE)

where `MAX` is the maximum possible pixel value (e.g., 255 for 8-bit images).

Intuitively:
- Lower MSE → Higher PSNR  
- Higher PSNR → Better reconstruction quality

PSNR is commonly adopted in **lossy image and video codecs** because it provides a normalized and interpretable quality metric, enabling fair comparisons across different codecs and bitrates.

Typical PSNR values for 8-bit images and video are:
- **< 25 dB**: poor quality, visible artifacts  
- **25–30 dB**: acceptable quality  
- **30–35 dB**: good quality  
- **> 35 dB**: high quality, differences barely noticeable

### Occlusions


Occlusions occur when parts of the scene become visible or invisible between consecutive frames due to object or camera motion. In these regions, no valid pixel correspondence exists between frames.

This is a problem for motion-based warping, since the warped frame relies on optical flow to map pixels from a previous frame. When occlusions are present, the motion compensation cannot correctly predict these areas, leading to visual artifacts and reconstruction errors. For this reason, occluded regions must be handled separately, typically through residual information or refinement networks.

# 2. Codec Architecture

Almost every video codec follows this paradigm. Starting from a frame $x_t$ at time $t$, the motion with respect to the previous frame $x_{t-1}$ (stored in a buffer) is estimated. The resulting motion field is then compressed, producing the first latent representation $m_t$.

After decoding $m_t$, the decompressed motion is used to warp the previous frame $x_{t-1}$, obtaining a motion-compensated prediction $\bar{x}_t$. The residual error is computed as the difference between the ground-truth frame $x_t$ and the predicted frame $\bar{x}_t$, and subsequently compressed into a second latent representation $y_t$.

Instead of transmitting the full frame $x_t$, only the two latent representations $(m_t, y_t)$ are sent and decoded using the previously reconstructed frame, resulting in a significant reduction in bitrate. The **DVC paper** embrace this approach with the following architecture:

<p align="center">
  <img src="images/dvc.png" alt="dvc" width="50%">
</p>

Here they use five trained neural networks:
 - **Optical Flow NET**: given the two consecutive frames in input, it computes the flow map between them.
 - **MV Encoder NET**: a VAE to compress and reconstruct the flow map.
 - **Motion Compensation NET**: a network that, taken the previous frame and the decompressed flow, outputs a warped prediction.
 - **Residual Encoder NET**: a VAE to compress and reconstruct the residual error.
 - **Bit Rate Estimation NET**: an entropy model that "predict" the bit rate of the VAEs.

## DVC's architecture problems

### No temporal context

As shown in following studies on this topic, such as **[M-LVC: Multiple Frames Prediction for Learned Video Compression](https://arxiv.org/pdf/2004.10290)** [2], temporal context can be a game-changing factor in handling occlusions. Previous reconstructed frames may contain information that is not available in the immediately preceding frame and can be exploited to improve motion compensation and residual prediction, especially in occluded or ambiguous regions.

### Deep-Network to obtain the warped prediction

DVC and several subsequent works rely on a deep neural network that, given the optical flow and the previous reconstructed frame (and in some cases the residual), predicts the current frame. In this setting, the network is required to reconstruct the frame almost from scratch. This approach can easily become a bottleneck during inference. Achieving high reconstruction quality often requires large networks with many parameters, which, combined with the computational cost of optical flow estimation, can significantly reduce compression and decompression speed, resulting in low frames per second (FPS).

# 2.1 Proposed Architecture

Our codec follows the classical motion–residual paradigm, while introducing several modern enhancements. The following image illustrates the overall architecture of the proposed codec:

<p align="center">
  <img src="images/net.png" alt="Our NET" width="80%">
</p>

Lets see all the components model in detail:

## Flow and Residual Encoder-Decoder VAEs 

The autoencoder architecture we've chosen is the one proposed in **[Variational Image Compression with a Scale Hyperprior](https://arxiv.org/pdf/1802.01436)** [3], and it is used to compress both the optical flow and the residual error.

In our setting, the decoder does not need to reconstruct an image from scratch, since it already relies on a strong prior given by the previously reconstructed frame $x_{t-1}$. This makes the hyperprior architecture particularly suitable: it is designed to preserve structural information while achieving high compression efficiency by modeling the spatial dependencies of the latent representation.

As highlighted in the original paper, the scale hyperprior allows a more accurate entropy model of the latent features, resulting in better rate–distortion performance. This property is especially beneficial in our case, where the information to be encoded (flow and residual) is already structured and correlated with the reference frame.

The implementation is based on the Scale Hyperprior model provided by [CompressAI - Model Documentation](https://interdigitalinc.github.io/CompressAI/models.html):


```python
class ScaleHyperprior(CompressionModel):
    def __init__(self, N, M, in_channels=2, out_channels=2, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

        # Analysis transform
        self.g_a = nn.Sequential(
            conv(in_channels, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        # Synthesis transform
        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, out_channels),
        )

        # Hyperprior analysis
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        # Hyperprior synthesis
        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        in_channels = state_dict["g_a.0.weight"].size(1)
        out_channels = state_dict["g_s.6.weight"].size(0)

        net = cls(N, M, in_channels=in_channels, out_channels=out_channels)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)

        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}
```

We slightly modified the original Scale Hyperprior implementation by feeding the absolute value of the latent representation to the hyperprior network (`z = h_a(|y|)`). 

This encourages the hyperprior to model the magnitude (scale) of the features rather than their sign, which is particularly suitable for optical flow and residual representations where the direction information is preserved in the main latent branch.

Additionally, the architecture was extended with configurable `in_channels` and `out_channels` parameters, since the model is used to compress both optical flow (2 channels) and residual error (3 channels). This makes the same Scale Hyperprior backbone adaptable to different types of inputs without altering its core structure.

## Post-proceessing NETs

Like **DVC** and other studies we use two post processing nets to refine both *Optical Flow* and *Residual*. The goal of the **_Flow Refine NET_** is to minimize the difference between the warped prediction and the ground-truth frame. The smaller this error is, the less information the residual needs to encode, directly reducing the number of bits required for its compression. The **_Residual Refine NET_** is a lightweight network made to denoise the decompressed residual before the final reconstruction. 

As discussed before, one limitation of several video codecs is the reliance on large CNNs designed to reconstruct the frame, the flow, or the residual almost entirely from scratch. In our architecture, we instead adopt two lightweight ResNet-based backbones.

ResNets are particularly suitable in this context because they are designed to learn residual corrections over an existing input rather than rebuilding the signal from the ground up. Thanks to their identity-mapping property, they provide stable refinements starting from a strong prior (the decompressed flow and residual), making them both efficient and well-suited for this task.

### Flow Refine NET

The `MotionRefineNET` refines the **decompressed optical flow** by predicting a small correction term.  
It takes as input the current flow estimate `flow_hat` (2 channels) concatenated with a **temporal context** of the last 4 reconstructed frames (like its done in **M-LVC**), for a total of 14 input channels.

The network is a lightweight ResNet-style model:
- a small convolutional **stem** to extract features,
- a stack of residual blocks (with larger dilation in the last blocks to enlarge the receptive field),
- a `delta_head` that predicts a **flow residual** `delta` (2 channels).

Finally, the refined flow is obtained as:
`flow_refined = flow_hat + delta`
meaning the model learns **small, stable updates** instead of reconstructing flow from scratch.

```python
class MotionRefineNET(nn.Module):
    def __init__(self, base=64, blocks=8, use_gn=True, use_gate=True):
        super().__init__()

        # Input: 2 (Flow) + 12 (History: t-1, t-2, t-3, t-4) = 14 channels
        in_ch = 14

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        body = []
        for i in range(blocks):
            dil = 1 if i < blocks - 2 else 2
            body.append(ResBlock(base, dilation=dil, use_gn=use_gn))

        self.body = nn.Sequential(*body)

        # Predict a correction (delta) for the flow
        self.delta_head = nn.Conv2d(base, 2, 3, padding=1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        self.use_gate = use_gate
        if use_gate:
            self.gate_head = nn.Conv2d(base, 1, 3, padding=1)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)

    def forward(self, flow_hat, history_4f):
        x = torch.cat([flow_hat, history_4f], dim=1)

        f = self.stem(x)
        f = self.body(f)

        delta = self.delta_head(f)

        if self.use_gate:
            gate = torch.sigmoid(self.gate_head(f))
            delta = gate * delta

        return flow_hat + delta
```

Group Normalization is used for training stability with small batch sizes (more in details in the training chapter), while the gating mechanism allows the network to apply corrections selectively only in regions where the motion estimation is unreliable.



## References

[1] G. Lu, W. Ouyang, D. Xu, X. Zhang, C. Cai, Z. Gao. **"DVC: An End-to-End Deep Video Compression Framework."** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019.

[2] J. Lin, D. Liu, H. Li, and F. Wu, **"M-LVC: Multiple Frames Prediction for Learned Video Compression,"** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.

[3] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, **"Variational Image Compression with a Scale Hyperprior,"** *International Conference on Learning Representations (ICLR)*, 2018. 
