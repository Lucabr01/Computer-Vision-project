# Optical Flow-based Neural Video Compression

1. [Introduction](#1-introduction)
   1. [Fundamentals](#11-fundamentals)

# 1. Introduction

Our project aims to build a *Low-bitrate Lossy Video Codec* that replicates and enhances the conventional video compression architecture. Traditional manually designed codecs rely on motion vectors and residual information to reconstruct non-key frames from previously decoded frames. The reason is that, as will be discussed later, these two pieces of information account for only a small fraction of the original frame size, yet they are sufficient to reconstruct it with high fidelity.

The paper implemented in this project is **[DVC: An End-to-End Deep Video Compression Framework](https://arxiv.org/pdf/1812.00101)** [1]. It represents one of the first neural-network-based approaches to video coding and introduces the general architecture adopted by many subsequent methods. Our goal was to understand this paradigm and extend it by addressing some of the limitations of the original DVC architecture, such as occlusions and the lack of explicit temporal context.

## 1.1 Fundamentals

Lets see all the main logical components for our codec:

### Optical Flow

Given two consecutive frames, the **Optical Flow** quantifies the motion of objects between them. In simple terms, optical flow describes the direction and magnitude (the speed) of the displacement of each pixel from the first frame to the second.

## References

[1] G. Lu, W. Ouyang, D. Xu, X. Zhang, C. Cai, Z. Gao. **"DVC: An End-to-End Deep Video Compression Framework."** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019.
