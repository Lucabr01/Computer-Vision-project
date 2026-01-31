# Optical Flow-based Neural Video Compression

1. [Introduction](#1-introduction)
   1. [Fundamentals](#11-fundamentals)

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

### Peak signal-to-noise ratio



## References

[1] G. Lu, W. Ouyang, D. Xu, X. Zhang, C. Cai, Z. Gao. **"DVC: An End-to-End Deep Video Compression Framework."** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019.
