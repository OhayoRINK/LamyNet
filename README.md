# LamyNet ❄️
> **A Lightweight Dual-Axis & Bandwise Spectral Gating Autoencoder for CSI Feedback Compression in Massive MIMO FDD Systems**
> | Lightweight Adaptive Multi-axis Yield Network for Massive MIMO CSI Feedback Compression

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Task: Massive MIMO CSI Compression](https://img.shields.io/badge/Task-CSI%20Feedback%20Compression-informational.svg)](https://github.com/OhayoRINK)

English | [한국어 (Korean)](README_KR.md)

---

## Overview

**LamyNet** (developed under the project codename `Lite-DPAS v5`) is an ultra-lightweight, high-fidelity deep learning framework proposed for improving **Channel State Information (CSI) feedback compression performance** in Massive MIMO FDD systems.

Accurate downlink CSI feedback from User Equipment (UE) to the Base Station (BS) is indispensable for closed-loop beamforming and spatial multiplexing. However, in massive MIMO systems with large antenna arrays and numerous OFDM subcarriers, feedback overhead scales prohibitively. 

While baseline architectures like CsiNet pioneer autoencoder-based feedback, they incur heavy parameter footprints on UEs, struggle with long-delay multipath propagation, and suffer severe performance degradation under aggressive compression ratios.

**LamyNet** overcomes these challenges by combining:
- **Dual-Axis Depthwise Convolutions** (capturing spatial and long-range delay features with low complexity)
- **Bandwise Spectral Gating** (adaptive frequency-domain reweighting via rFFT energy distribution)
- **Spatial Feature Preservation** ($16 \times 16$ spatial grid instead of premature $8 \times 8$ bottlenecks)
- **Learned PixelShuffle Synthesis** (sub-pixel upsampling eliminating checkerboard interpolation artifacts)
- **50% Parameter Reduction** compared to standard CsiNet while maintaining competitive reconstruction accuracy across 1000-epoch training.

---

## Model Architecture

![LamyNet Architecture](source/architecture_diagram.png)

```text
========================================================================================
[UE Encoder]
Input CSI [2, 32, 32] 
   │
   ├─► ConvNormAct (2 -> 8, 3x3)
   ├─► DualAxisLiteBlock (8 channels, 3x3 Spatial DW + 1x15 Delay DW)
   ├─► LearnedDownsample (Stride-2 Conv + DualAxis, 32x32 -> 16x16)
   ├─► BandwiseSpectralGate (rFFT on delay axis, 4-band adaptive gating)
   ├─► DualAxisLiteBlock (8 channels)
   ├─► Conv1x1 Channel Slimming (8 -> 4 channels)
   └─► Flatten [1024] ──► Linear Projection ──► Codeword [Dim = 2048 / CR]
                                                         │
====================================== Uplink Feedback Channel =========================
                                                         │
[BS Decoder]                                             ▼
Codeword [Dim = 2048 / CR] ──► Linear Expansion ──► Reshape [4, 16, 16]
   │
   ├─► DualAxisLiteBlock (4 channels)
   ├─► ConvNormAct (4 -> 8, 1x1)
   ├─► BandwiseSpectralGate (8 channels, 4-band delay spectrum reweighting)
   ├─► LearnedPixelShuffleUp (Sub-pixel 2x upsampling, 16x16 -> 32x32)
   ├─► Reconstruction Conv (8 -> 2, 3x3) + Sigmoid
   └─► Reconstructed CSI [2, 32, 32]
========================================================================================
```

---

## Key Innovations

### 1. Dual-Axis Depthwise Convolution (`DualAxisLiteBlock`)
Conventional 2D convolutions naively blend angle and delay axes with quadratic kernel parameter costs. LamyNet decouples feature extraction into two parallel depthwise branches:
- **Spatial Path**: $3 \times 3$ Depthwise Conv for localized angle-domain correlations.
- **Delay Path**: $1 \times 15$ Depthwise Conv capturing extended delay-axis multipath spreads across OFDM subcarriers.
- **Pointwise Fusion**: $1 \times 1$ Conv fuses the concatenated branches with residual connections.

```text
Input Feature Map
   ├── Depthwise Conv 3×3 (Spatial Path) ────┐
   └── Depthwise Conv 1×15 (Delay Path) ────┴──► Concat ──► 1×1 Conv ──► (+) Residual ──► SiLU
```

### 2. Bandwise Spectral Gating (`BandwiseSpectralGate`)
Unlike spatial or channel attention mechanisms that operate purely in time/delay space, the **Bandwise Spectral Gate** leverages fast real-valued Fourier transforms (`torch.fft.rfft`) along the delay axis:
1. Calculates delay-domain power spectrum magnitudes.
2. Pools energy into 4 discrete frequency bands via `adaptive_avg_pool1d`.
3. Learns non-linear band-importance weights through a light MLP.
4. Dynamically scales corresponding frequency bins before inverse transform (`irfft`), filtering noise and emphasizing dominant multipath clusters.

### 3. Spatial Resolution Preservation ($16 \times 16$)
Earlier prototypes (v1–v3) compressed features directly down to an $8 \times 8$ grid, suffering significant structural detail loss. LamyNet maintains a high-resolution $16 \times 16$ spatial grid while slimming the channel count to $4$. This yields a compact $4 \times 16 \times 16 = 1024$ latent representation before feedback projection, drastically reducing parameter count in fully connected layers while protecting spatial information.

### 4. Sub-Pixel Convolution Synthesis (`LearnedPixelShuffleUp`)
Instead of fixed bilinear interpolation or parameter-heavy transposed convolutions, LamyNet employs **PixelShuffle** with learned depthwise refinement:
- Expands channel depth ($8 \rightarrow 32$) and rearranges spatial dimensions ($16 \times 16 \rightarrow 32 \times 32$).
- Completely avoids checkerboard artifacts common in deconvolutional decoders.

---

## Experimental Results

All experiments were conducted on the standard **COST2100** channel dataset (Indoor at 5.3 GHz and Outdoor at 300 MHz). Models were trained from scratch on a single **NVIDIA RTX 3080 (10 GB)** for **1,000 full epochs** with NMSE loss, Adam (lr 1e-3), batch 200, `ReduceLROnPlateau` scheduling and seed 42. The checkpoint with the lowest **validation** NMSE is kept; the test split is used only for the final evaluation. The checkpoints are available in the [`checkpoints` release](https://github.com/OhayoRINK/LamyNet/releases/tag/checkpoints).

### 1. Overall Performance Table (1,000 Epochs)

| Scenario | Compression Ratio (CR) | Encoded Dim | Best Epoch | Val NMSE (dB) | **Test NMSE (dB)** | **Test Cosine Sim ($\rho$)** | Parameters | FLOPs (MACs) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Indoor** | **1/4** | 512 | 991 | **-20.425** | **-20.405** | **0.99273** | 1,057,188 | 5.90M (2.95M) |
| **Indoor** | **1/16** | 128 | 993 | -10.236 | -10.233 | 0.95093 | 270,372 | 4.33M (2.16M) |
| **Indoor** | **1/32** | 64 | 981 | -8.800 | -8.780 | 0.93192 | 139,236 | 4.07M (2.03M) |
| **Indoor** | **1/64** | 32 | 995 | -7.043 | -7.017 | 0.89869 | **73,668** | **3.94M (1.97M)** |
| **Outdoor** | **1/4** | 512 | 1000 | **-9.116** | **-9.116** | **0.91813** | 1,057,188 | 5.90M (2.95M) |
| **Outdoor** | **1/16** | 128 | 987 | -5.271 | -5.271 | 0.83122 | 270,372 | 4.33M (2.16M) |
| **Outdoor** | **1/32** | 64 | 994 | -3.546 | -3.546 | 0.72991 | 139,236 | 4.07M (2.03M) |
| **Outdoor** | **1/64** | 32 | 999 | -2.301 | -2.301 | 0.63713 | **73,668** | **3.94M (1.97M)** |

> **Evaluation Metrics**:
> - **NMSE (dB)**: $10 \log_{10} \mathbb{E} \left[ \frac{\|\mathbf{H} - \widehat{\mathbf{H}}\|_2^2}{\|\mathbf{H}\|_2^2} \right]$ (with normalized DC 0.5 offset centered)
> - **Cosine Similarity ($\rho$)**: CsiNet definition, $\mathbb{E} \left[ \frac{1}{N_s} \sum_{n=1}^{N_s} \frac{|\hat{\mathbf{h}}_n^H \mathbf{h}_n|}{\|\hat{\mathbf{h}}_n\|_2 \|\mathbf{h}_n\|_2} \right]$ over the first $N_s = 125$ subcarriers (32 delay taps zero-padded to a 257-point FFT, compared with the raw frequency-domain channel)

---

### 2. Parameter Efficiency vs. CsiNet Baseline

By replacing dense $32 \times 32$ fully connected bottlenecks with a $16 \times 16 \times 4$ intermediate manifold and depthwise operators, LamyNet cuts parameter costs by approximately **50%**:

| Compression Ratio | CsiNet Baseline Params | **LamyNet Params** | **Parameter Reduction** |
|:---:|:---:|:---:|:---:|
| **1/4** | 2,103,116 (2.10M) | **1,057,188 (1.06M)** | **-49.7%** |
| **1/16** | 529,868 (0.53M) | **270,372 (0.27M)** | **-49.0%** |
| **1/32** | 267,660 (0.27M) | **139,236 (0.14M)** | **-48.0%** |
| **1/64** | 136,556 (0.14M) | **73,668 (0.07M)** | **-46.1%** |

---

### 3. Training Convergence Curves (1,000 Epochs)

#### Indoor Scenario Validation NMSE
![Indoor Validation NMSE](source/chart_indoor.png)

#### Outdoor Scenario Validation NMSE
![Outdoor Validation NMSE](source/chart_outdoor.png)

#### Side-by-Side Convergence Comparison
![Combined Convergence](source/chart_convergence_combined.png)

#### Test NMSE Bar Comparison across Compression Ratios
![Test NMSE Bar Comparison](source/chart_test_nmse_bars.png)

---

## Directory Structure

```text
L_D/
├── source/                      # Generated evaluation charts & architecture diagram
│   ├── architecture_diagram.png
│   ├── chart_indoor.png
│   ├── chart_outdoor.png
│   ├── chart_convergence_combined.png
│   ├── chart_test_nmse_bars.png
│   ├── evaluation_results.json
│   ├── experiment_results.json  # Test NMSE / rho of all RTX 3080 runs (also .csv)
│   ├── latency_onnx_results.json # ONNX Runtime CPU inference time
│   └── deploy_equivalence.json  # Deployed vs. trained model check
├── src/
│   ├── models/
│   │   ├── lite_dpas.py         # LamyNet (LiteDPASCsiNetV5) architecture
│   │   ├── csinet.py            # Reference CsiNet baseline
│   │   └── lite_dpas_1736.py    # Experimental prototype variants
│   ├── cost2100.py              # PyTorch Dataset loader for COST2100 .mat files
│   ├── metrics.py               # Complex-equivalent NMSE and rho evaluation
│   ├── profile_lite_dpas.py     # MACs, parameter counter, and inference latency profiler
│   ├── train_baseline.py        # Core training engine with scheduler & checkpointing
│   ├── train_lite_dpas.py       # Training entrypoint (LamyNet, ablations, CsiNet/CRNet/CLNet/TransNet)
│   ├── train_lite_dpas_outdoor.py # Outdoor training entrypoint
│   ├── upstream_models.py       # Authors' CRNet / CLNet / TransNet loader
│   ├── evaluate_runs.py         # Test NMSE / rho of every run
│   ├── deploy.py                # Deployment conversion (BN folding, DFT-matmul spectral gate)
│   ├── check_deploy_equivalence.py # Deployed vs. trained model check
│   └── profile_latency_onnx.py  # ONNX Runtime CPU inference time
├── experiments/
│   └── run_queue.py             # Experiment queue (ablation, baseline reproduction, LamyNet runs)
├── runs/                        # 1000-epoch checkpoints and training logs
│   ├── litedpas_v5_indoor_cr4_1000ep/
│   ├── litedpas_v5_indoor_cr16_1000ep/
│   ├── litedpas_v5_indoor_cr32_1000ep/
│   ├── litedpas_v5_indoor_cr64_1000ep/
│   ├── litedpas_v5_outdoor_cr4_1000ep/
│   ├── litedpas_v5_outdoor_cr16_1000ep/
│   ├── litedpas_v5_outdoor_cr32_1000ep/
│   └── litedpas_v5_outdoor_cr64_1000ep/
├── notebooks/                   # Jupyter notebooks for experiment exploration
├── evaluate_and_plot.py         # Evaluation and automated plot generator
├── draw_diagram.py              # Architecture diagram generator script
├── README.md                    # English documentation
└── README_KR.md                 # Korean documentation
```

---

## Getting Started

### Prerequisites

```bash
git clone https://github.com/OhayoRINK/LamyNet.git
cd LamyNet
pip install torch torchvision numpy scipy matplotlib
```

### Dataset Setup

Download the standard **COST2100** channel matrices (`DATA_Htrainin.mat`, `DATA_Hvalin.mat`, `DATA_Htestin.mat`, etc.) and place them inside a `Data/` directory:

```text
Data/
├── DATA_Htrainin.mat
├── DATA_Hvalin.mat
├── DATA_Htestin.mat
├── DATA_Htrainout.mat
├── DATA_Hvalout.mat
└── DATA_Htestout.mat
```

### Training

To train LamyNet (Lite-DPAS v5) from scratch:

```bash
# Indoor Scenario, CR = 1/32, 1000 Epochs
python src/train_lite_dpas.py \
  --compression-ratio 32 \
  --model litedpas_v5 \
  --epochs 1000 \
  --batch-size 200 \
  --loss-mode nmse \
  --scheduler-patience 25

# Outdoor Scenario, CR = 1/16, 1000 Epochs
python src/train_lite_dpas_outdoor.py \
  --compression-ratio 16 \
  --model litedpas_v5 \
  --epochs 1000 \
  --batch-size 200 \
  --loss-mode nmse \
  --scheduler-patience 25
```

### Evaluation & Profiling

To evaluate checkpoint performance and measure MACs/latency:

```bash
# Generate complete evaluation metrics and plots across all 8 runs:
python evaluate_and_plot.py

# Profile individual checkpoint:
python src/profile_lite_dpas.py \
  --checkpoint runs/litedpas_v5_indoor_cr4_1000ep/best.pt \
  --samples 512

# Test NMSE / rho of every run under runs/ -> source/experiment_results.json (.csv)
python -m src.evaluate_runs --data-dir Data

# Deployment conversion check (BN folding, DFT-matmul gate) -> source/deploy_equivalence.json
python -m src.check_deploy_equivalence --data-dir Data

# ONNX Runtime CPU inference time (batch 1, one thread) -> source/latency_onnx_results.json
# LAMYNET_BASELINES points to a folder holding the authors' CRNet/, CLNet/ and TransNet/ repositories
LAMYNET_BASELINES=/path/to/baselines python -m src.profile_latency_onnx
```


This work builds upon the foundational principles of [CsiNet](https://ieeexplore.ieee.org/document/8400482), [PekoraNet](https://github.com/OhayoRINK/PekoraNet), and [DPAS-CsiNet](https://github.com/OhayoRINK/DPAS-CsiNet).
