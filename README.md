# 🎨 DDPM — Denoising Diffusion Probabilistic Model on CIFAR-10

Lab Assignment: Media Generation via Deep Learning Models

## Overview

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch in PyTorch to generate 32×32 colour images trained on CIFAR-10.

| Property | Value |
|----------|-------|
| Model | DDPM with U-Net backbone |
| Dataset | CIFAR-10 (50 000 images, 10 classes) |
| Resolution | 32 × 32 × 3 (RGB) |
| Timesteps | T = 1000 |
| Parameters | ~7 M |

## How It Works

1. **Forward process** — gradually add Gaussian noise over 1 000 steps until the image becomes pure noise  
2. **Reverse process** — train a U-Net to predict and remove the noise step by step  
3. **Sampling** — start from pure noise and apply the learned denoiser T times to produce a new image

## Architecture

```
Input (3×32×32)
    ↓
InitConv → [Encoder: 3 levels with ResBlocks + Attention]
    ↓
Bottleneck (ResBlock → Attention → ResBlock)
    ↓
[Decoder: 3 levels with skip connections]
    ↓
Output Conv → Predicted noise (3×32×32)
```

Time step `t` is embedded via **sinusoidal positional encodings** and injected into every ResBlock.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Open and run `diffusion_cifar10.ipynb` in Jupyter:

```bash
jupyter notebook diffusion_cifar10.ipynb
```

Generated images and training curves are saved to `./outputs/`.

## Results

| Epoch | Sample |
|-------|--------|
| 1 | Mostly noise with faint structure |
| 25 | Coarse shapes visible |
| 50 | Recognisable objects, some blurriness |
| 200+ | Sharp, diverse CIFAR-10-like images |

> **Tip:** For best results, train for 200+ epochs on a GPU. Colab/Kaggle free GPUs work well.

## References

- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.* [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Dhariwal & Nichol (2021). *Diffusion Models Beat GANs on Image Synthesis.* [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
