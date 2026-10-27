# LamyNet ❄️
> **Massive MIMO FDD 시스템의 CSI 피드백 압축 성능 향상을 위한 초경량 듀얼축 합성곱 및 대역별 스펙트럴 게이팅 오토인코더**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Task: Massive MIMO CSI Compression](https://img.shields.io/badge/Task-CSI%20Feedback%20Compression-informational.svg)](https://github.com/OhayoRINK)

[English](README.md) | 한국어 (Korean)

---

## 개요 (Overview)

**LamyNet** (프로젝트 개발 코드명 `Lite-DPAS v5`)은 **Massive MIMO FDD 시스템에서 CSI(채널 상태 정보) 피드백 압축 성능을 획기적으로 개선**하기 위해 고안된 초경량 고충실도 딥러닝 프레임워크입니다.

기지국(BS)의 폐루프 빔포밍 및 공간 다중화 성능을 극대화하기 위해서는 사용자 단말(UE)이 하향링크 CSI를 기지국으로 정확하게 피드백해야 합니다. 그러나 대규모 안테나와 수많은 OFDM 부반송파를 사용하는 Massive MIMO 환경에서는 피드백 오버헤드가 기하급수적으로 증가합니다.

기존 CsiNet 등 초기 오토인코더 모델은 단말(UE) 측 완전연결계층(FC) 파라미터 부담이 크고, 긴 지연시간을 갖는 다중경로 성분을 효과적으로 포착하지 못하며, 고압축률(CR=1/32, 1/64) 환경에서 심각한 성능 저하가 발생하는 한계가 있었습니다.

**LamyNet**은 이러한 한계를 극복하기 위해 다음과 같은 핵심 기술을 도입했습니다:
- **듀얼축 깊이별 분리 합성곱 (Dual-Axis Depthwise Separable Conv)**: 각도 축($3 \times 3$)과 롱-딜레이 축($1 \times 15$)을 독립 추출하여 파라미터 및 연산량 대폭 절감
- **대역별 스펙트럴 게이팅 (Bandwise Spectral Gating)**: 딜레이 축 고속 실수 푸리에 변환(rFFT) 기반의 4-Band 주파수 에너지 적응형 가중치 재조정
- **공간 해상도 보존 ($16 \times 16$)**: 과도한 병목 축소($8 \times 8$)를 지양하고 $16 \times 16 \times 4$ 구조를 유지하여 핵심 채널 공간 구조 보호
- **학습형 서브픽셀 합성 (PixelShuffle Synthesis)**: 단순 Bilinear 보간 대신 PixelShuffle 업샘플링을 적용해 격자 왜곡 없이 고주파 채널 성분 정밀 복원
- **CsiNet 대비 파라미터 약 50% 절감**: 1000 Epoch 장기 학습을 통해 안정적인 수렴과 우수한 복원 성능 입증

---

## 모델 아키텍처 (Architecture)

![LamyNet Architecture](source/architecture_diagram.png)

```text
========================================================================================
[UE 인코더 (송신단)]
입력 CSI [2, 32, 32] 
   │
   ├─► ConvNormAct (2 -> 8 채널, 3x3)
   ├─► DualAxisLiteBlock (8 채널, 3x3 공간 DW + 1x15 딜레이 DW)
   ├─► LearnedDownsample (Stride-2 다운샘플링, 32x32 -> 16x16)
   ├─► BandwiseSpectralGate (딜레이 축 rFFT, 4개 대역 적응형 게이팅)
   ├─► DualAxisLiteBlock (8 채널)
   ├─► Conv1x1 채널 슬림화 (8 -> 4 채널)
   └─► Flatten [1024] ──► 선형 투영(Linear FC) ──► 피드백 코드워드 [차원 = 2048 / CR]
                                                               │
====================================== 상향링크 피드백 채널 ===========================
                                                               │
[BS 디코더 (수신단)]                                            ▼
피드백 코드워드 [차원 = 2048 / CR] ──► 선형 확장(Linear FC) ──► Reshape [4, 16, 16]
   │
   ├─► DualAxisLiteBlock (4 채널)
   ├─► ConvNormAct (4 -> 8 채널, 1x1)
   ├─► BandwiseSpectralGate (8 채널, 4대역 주파수 재조정)
   ├─► LearnedPixelShuffleUp (서브픽셀 2배 업샘플링, 16x16 -> 32x32)
   ├─► 최종 복원 Conv (8 -> 2 채널, 3x3) + Sigmoid
   └─► 복원된 CSI [2, 32, 32]
========================================================================================
```

---

## 핵심 혁신 모듈 (Key Innovations)

### 1. 듀얼축 깊이별 합성곱 (`DualAxisLiteBlock`)
일반 2D 합성곱은 각도(Spatial)와 지연(Delay) 축을 한 번에 결합하여 파라미터가 크게 낭비됩니다. LamyNet은 두 축의 물리적 특성을 고려하여 깊이별 합성곱(Depthwise Conv)을 병렬로 분리했습니다:
- **공간 경로 (Spatial Path)**: 국소 각도 도메인 상관관계를 포착하는 $3 \times 3$ Depthwise Conv
- **지연 경로 (Delay Path)**: OFDM 다중경로 지연 확산을 포착하는 $1 \times 15$ Long-Delay Depthwise Conv
- **점별 융합 (Pointwise Fusion)**: 두 경로를 결합 후 $1 \times 1$ Conv와 잔차 연결(Residual Connection)로 융합

```text
입력 피처맵
   ├── Depthwise Conv 3×3 (공간 축 경로) ─────┐
   └── Depthwise Conv 1×15 (지연 축 경로) ────┴──► Concat ──► 1×1 Conv ──► (+) 잔차 ──► SiLU
```

### 2. 대역별 스펙트럴 게이팅 (`BandwiseSpectralGate`)
단순 공간/채널 어텐션과 달리, 딜레이 축 고속 실수 푸리에 변환(`torch.fft.rfft`)을 직접 수행합니다:
1. 딜레이 도메인의 스펙트럼 크기(Magnitude)를 산출합니다.
2. `adaptive_avg_pool1d`를 통해 4개의 주파수 대역으로 에너지를 집약합니다.
3. 경량 MLP를 통해 대역별 적응형 가중치(Gating weight)를 생성합니다.
4. 역변환(`irfft`) 전에 해당 주파수 bin에 게이트를 곱해줌으로써 노이즈를 억제하고 유효 다중경로 성분을 강조합니다.

### 3. 공간 해상도 보존 메커니즘 ($16 \times 16$)
이전 실험 모델들은 피처맵을 $8 \times 8$까지 무리하게 축소하여 미세 구조가 소실되는 문제가 있었습니다. LamyNet은 공간 해상도를 $16 \times 16$으로 온전히 유지하면서 채널 수를 4개로 슬림화했습니다. 이를 통해 $4 \times 16 \times 16 = 1024$의 컴팩트한 잠재 표현을 형성하여 단말의 완전연결계층(FC) 파라미터를 획기적으로 줄였습니다.

### 4. 서브픽셀 픽셀셔플 합성 (`LearnedPixelShuffleUp`)
단순 보간(Bilinear)이나 무거운 전치합성곱(ConvTranspose2d) 대신 **PixelShuffle** 기법을 적용했습니다:
- 채널을 4배($8 \rightarrow 32$)로 확장한 뒤 공간 해상도를 2배($16 \times 16 \rightarrow 32 \times 32$)로 재배열
- 디컨볼루션 특유의 바둑판 격자 왜곡(Checkerboard Artifact)을 원천 차단하고 세밀한 CSI 고주파수를 선명하게 복원합니다.

---

## 실험 결과 (Experimental Results)

표준 **COST2100** 채널 데이터셋(실내 5.3 GHz, 실외 300 MHz)을 사용하였습니다. 모든 모델은 **NVIDIA RTX 3080(10 GB)** 한 장에서 NMSE 손실, Adam(학습률 1e-3), 배치 200, `ReduceLROnPlateau`, 시드 42로 **1,000 Epoch** 동안 처음부터 학습하였습니다. 체크포인트는 **검증** NMSE가 가장 낮은 epoch를 사용하며, 시험 데이터는 최종 평가에만 사용합니다. 학습된 가중치는 [`checkpoints` 릴리즈](https://github.com/OhayoRINK/LamyNet/releases/tag/checkpoints)에서 받을 수 있습니다.

### 1. 종합 성능 결과표 (1,000 Epoch 완주)

| 시나리오 | 압축률 (CR) | 잠재 차원 | 최적 Epoch | 검증 NMSE (dB) | **테스트 NMSE (dB)** | **테스트 Cosine Sim ($\rho$)** | 파라미터 수 | 연산량 (FLOPs / MACs) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **실내 (Indoor)** | **1/4** | 512 | 991 | **-20.425** | **-20.405** | **0.99273** | 1,057,188 (1.06M) | 5.90M (2.95M) |
| **실내 (Indoor)** | **1/16** | 128 | 993 | -10.236 | -10.233 | 0.95093 | 270,372 (0.27M) | 4.33M (2.16M) |
| **실내 (Indoor)** | **1/32** | 64 | 981 | -8.800 | -8.780 | 0.93192 | 139,236 (0.14M) | 4.07M (2.03M) |
| **실내 (Indoor)** | **1/64** | 32 | 995 | -7.043 | -7.017 | 0.89869 | **73,668 (0.07M)** | **3.94M (1.97M)** |
| **실외 (Outdoor)** | **1/4** | 512 | 1000 | **-9.116** | **-9.116** | **0.91813** | 1,057,188 (1.06M) | 5.90M (2.95M) |
| **실외 (Outdoor)** | **1/16** | 128 | 987 | -5.271 | -5.271 | 0.83122 | 270,372 (0.27M) | 4.33M (2.16M) |
| **실외 (Outdoor)** | **1/32** | 64 | 994 | -3.546 | -3.546 | 0.72991 | 139,236 (0.14M) | 4.07M (2.03M) |
| **실외 (Outdoor)** | **1/64** | 32 | 999 | -2.301 | -2.301 | 0.63713 | **73,668 (0.07M)** | **3.94M (1.97M)** |

> **평가 지표**:
> - **NMSE (dB)**: $10 \log_{10} \mathbb{E} \left[ \frac{\|\mathbf{H} - \widehat{\mathbf{H}}\|_2^2}{\|\mathbf{H}\|_2^2} \right]$ (정규화 오프셋 0.5 DC 제거 후 계산)
> - **코사인 유사도 ($\rho$)**: CsiNet 정의, $\mathbb{E} \left[ \frac{1}{N_s} \sum_{n=1}^{N_s} \frac{|\hat{\mathbf{h}}_n^H \mathbf{h}_n|}{\|\hat{\mathbf{h}}_n\|_2 \|\mathbf{h}_n\|_2} \right]$, 앞쪽 $N_s = 125$개 부반송파 기준 (지연 탭 32개를 257점 FFT로 제로 패딩한 뒤 원시 주파수 영역 채널과 비교)

---

### 2. 베이스라인(CsiNet) 대비 파라미터 경량화 비교

LamyNet은 $16 \times 16 \times 4$ 중간 잠재 그리드와 깊이별 합성곱 설계를 통해, CsiNet 대비 **파라미터 수를 약 50% 절감**하였습니다:

| 압축률 (CR) | CsiNet 기준 파라미터 | **LamyNet 파라미터** | **파라미터 절감률** |
|:---:|:---:|:---:|:---:|
| **1/4** | 2,103,116 (2.10M) | **1,057,188 (1.06M)** | **-49.7%** |
| **1/16** | 529,868 (0.53M) | **270,372 (0.27M)** | **-49.0%** |
| **1/32** | 267,660 (0.27M) | **139,236 (0.14M)** | **-48.0%** |
| **1/64** | 136,556 (0.14M) | **73,668 (0.07M)** | **-46.1%** |

---

### 3. 학습 수렴 곡선 (1,000 Epochs)

#### 실내 시나리오 (Indoor) 검증 NMSE 수렴 곡선
![Indoor Validation NMSE](source/chart_indoor.png)

#### 실외 시나리오 (Outdoor) 검증 NMSE 수렴 곡선
![Outdoor Validation NMSE](source/chart_outdoor.png)

#### 종합 비교 수렴 곡선
![Combined Convergence](source/chart_convergence_combined.png)

#### 압축률별 테스트 NMSE 성능 비교 바차트
![Test NMSE Bar Comparison](source/chart_test_nmse_bars.png)

---

## 디렉터리 구조 (Directory Structure)

```text
L_D/
├── source/                      # 생성된 평가 그래프 및 아키텍처 다이어그램
│   ├── architecture_diagram.png
│   ├── chart_indoor.png
│   ├── chart_outdoor.png
│   ├── chart_convergence_combined.png
│   ├── chart_test_nmse_bars.png
│   ├── evaluation_results.json
│   ├── experiment_results.json  # RTX 3080 학습 결과 전체의 시험 NMSE / rho (.csv 포함)
│   ├── latency_onnx_results.json # ONNX Runtime CPU 추론 시간
│   └── deploy_equivalence.json  # 배포 변환 모델과 학습 모델 출력 비교
├── src/
│   ├── models/
│   │   ├── lite_dpas.py         # LamyNet (LiteDPASCsiNetV5) 핵심 아키텍처
│   │   ├── csinet.py            # 기준 CsiNet 베이스라인 모델
│   │   └── lite_dpas_1736.py    # 실험 프로토타입 모델군
│   ├── cost2100.py              # COST2100 .mat 데이터셋 PyTorch 로더
│   ├── metrics.py               # 복소수 환산 NMSE 및 코사인 유사도 평가 지표
│   ├── profile_lite_dpas.py     # MACs, 파라미터 수, 추론 지연시간 프로파일러
│   ├── train_baseline.py        # 학습/검증 엔진 (스케줄러, 체크포인트 저장)
│   ├── train_lite_dpas.py       # 학습 CLI (LamyNet, 소거 실험, CsiNet/CRNet/CLNet/TransNet)
│   ├── train_lite_dpas_outdoor.py # 실외 시나리오 학습 CLI 엔트리포인트
│   ├── upstream_models.py       # 저자 공개 CRNet / CLNet / TransNet 불러오기
│   ├── evaluate_runs.py         # 전체 run의 시험 NMSE / rho 집계
│   ├── deploy.py                # 배포 변환 (BN 접기, 스펙트럴 게이트 DFT 행렬곱)
│   ├── check_deploy_equivalence.py # 배포 변환 모델과 학습 모델 출력 비교
│   └── profile_latency_onnx.py  # ONNX Runtime CPU 추론 시간 측정
├── experiments/
│   └── run_queue.py             # 실험 큐 (소거 실험, 기준 모델 재현, LamyNet 학습)
├── runs/                        # 1000 Epoch 학습 체크포인트 및 로그
│   ├── litedpas_v5_indoor_cr4_1000ep/
│   ├── litedpas_v5_indoor_cr16_1000ep/
│   ├── litedpas_v5_indoor_cr32_1000ep/
│   ├── litedpas_v5_indoor_cr64_1000ep/
│   ├── litedpas_v5_outdoor_cr4_1000ep/
│   ├── litedpas_v5_outdoor_cr16_1000ep/
│   ├── litedpas_v5_outdoor_cr32_1000ep/
│   └── litedpas_v5_outdoor_cr64_1000ep/
├── notebooks/                   # 실험 분석용 Jupyter 노트북
├── evaluate_and_plot.py         # 결과 평가 및 고화질 그래프 자동 생성 스크립트
├── draw_diagram.py              # 아키텍처 블록 다이어그램 생성 스크립트
├── README.md                    # 영문 공식 문서
└── README_KR.md                 # 한국어 공식 문서
```

---

## 실행 가이드 (Getting Started)

### 환경 구성

```bash
git clone https://github.com/OhayoRINK/LamyNet.git
cd LamyNet
pip install torch torchvision numpy scipy matplotlib
```

### 데이터셋 준비

표준 **COST2100** 채널 행렬 파일(`DATA_Htrainin.mat`, `DATA_Hvalin.mat`, `DATA_Htestin.mat` 등)을 `Data/` 디렉터리에 배치합니다:

```text
Data/
├── DATA_Htrainin.mat
├── DATA_Hvalin.mat
├── DATA_Htestin.mat
├── DATA_Htrainout.mat
├── DATA_Hvalout.mat
└── DATA_Htestout.mat
```

### 모델 학습 (Training)

스크래치부터 1000 Epoch 학습을 진행할 때:

```bash
# 실내 시나리오 (Indoor), CR = 1/32, 1000 Epochs
python src/train_lite_dpas.py \
  --compression-ratio 32 \
  --model litedpas_v5 \
  --epochs 1000 \
  --batch-size 200 \
  --loss-mode nmse \
  --scheduler-patience 25

# 실외 시나리오 (Outdoor), CR = 1/16, 1000 Epochs
python src/train_lite_dpas_outdoor.py \
  --compression-ratio 16 \
  --model litedpas_v5 \
  --epochs 1000 \
  --batch-size 200 \
  --loss-mode nmse \
  --scheduler-patience 25
```

### 평가 및 프로파일링 (Evaluation & Profiling)

저장된 체크포인트를 평가하고 그래프 및 프로파일링을 실행합니다:

```bash
# 8개 런 전체에 대한 종합 평가 및 그래프 생성:
python evaluate_and_plot.py

# 개별 체크포인트 연산량(MACs) 및 지연시간 프로파일링:
python src/profile_lite_dpas.py \
  --checkpoint runs/litedpas_v5_indoor_cr4_1000ep/best.pt \
  --samples 512

# runs/ 아래 전체 run의 시험 NMSE / rho -> source/experiment_results.json (.csv)
python -m src.evaluate_runs --data-dir Data

# 배포 변환(BN 접기, DFT 행렬곱 게이트) 검증 -> source/deploy_equivalence.json
python -m src.check_deploy_equivalence --data-dir Data

# ONNX Runtime CPU 추론 시간 (배치 1, 스레드 1) -> source/latency_onnx_results.json
# LAMYNET_BASELINES는 저자 공개 CRNet/, CLNet/, TransNet/ 저장소가 있는 폴더
LAMYNET_BASELINES=/path/to/baselines python -m src.profile_latency_onnx
```

본 연구는 [CsiNet](https://ieeexplore.ieee.org/document/8400482), [PekoraNet](https://github.com/OhayoRINK/PekoraNet), [DPAS-CsiNet](https://github.com/OhayoRINK/DPAS-CsiNet)의 연구적 성과를 계승 및 확장하였습니다.
