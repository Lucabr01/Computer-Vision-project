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

## References

[1] G. Lu, W. Ouyang, D. Xu, X. Zhang, C. Cai, Z. Gao. **"DVC: An End-to-End Deep Video Compression Framework."** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019.

[2] J. Lin, D. Liu, H. Li, and F. Wu, **"M-LVC: Multiple Frames Prediction for Learned Video Compression,"** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.
