# FSL-EAC: Few-Shot Learning for Environmental Audio Classification



A comprehensive study of Few-Shot Learning methods for environmental audio classification on the DCASE2018 Task 5 dataset, comparing Transductive Fine-Tuning with Prototypical Networks, and evaluating SSAMBA (Mamba) vs PANNS (CNN) architectures.

## 🎯 Key Results

| Method | Model | Feature Dim | Accuracy |
|--------|-------|-------------|----------|
| **Transductive FT** | SSAMBA | 768D | **75.87%** |
| **PANNS (Best)** | CNN | 2048D | **87.02%** |
| SSAMBA 2048D | Mamba | 2048D | **82.18%** |
| Prototypical | SSAMBA | 768D | 64.00% |

**Key Findings**:
- ✅ Transductive FT outperforms Prototypical Networks by **+11.87%**
- ✅ Data augmentation (SpecAugment 4×) provides **+7.78%** improvement
- ✅ PANNS (CNN) achieves **87.02%**, outperforming SSAMBA (Mamba) by ~1-5%
- ✅ Multi-layer feature fusion improves SSAMBA from 75.87% to 82.18%

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Experiments](#experiments)
- [Reproduction](#reproduction)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## 🔍 Overview

### Research Questions

This project investigates:
1. **Fine-Tuning Strategies**: Does Transductive FT outperform Prototypical Networks?
2. **Data Augmentation**: How much does SpecAugment help in few-shot scenarios?
3. **Architecture Comparison**: SSAMBA (Mamba) vs PANNS (CNN) for audio spectrograms?
4. **Dimension Ablation**: Is higher dimensionality (2048D vs 768D) beneficial?
5. **Multi-Layer Features**: Can layer fusion improve SSAMBA performance?

### Method: Transductive Fine-Tuning

Unlike traditional meta-learning approaches, our method leverages both support and query sets:

```
For each episode:
  1. Extract features from support samples (with SpecAugment 4×)
  2. Initialize classifier with prototypes
  3. Fine-tune for 5 steps:
     - Support CE Loss: Standard classification loss
     - Query Entropy Loss: Encourages confident predictions (unlabeled)
  4. Evaluate on query set
```

**Key Innovation**: Query entropy regularization provides additional learning signal from unlabeled data distribution.

---

## 🛠️ Installation

### Requirements

- Python 3.12
- CUDA 12.8 


### Step 1: Clone Repository

```bash
git clone https://github.com/Gostyan/FSL-EAC.git
cd FSL-EAC
```

### Step 2: Create Conda Environment

```bash
conda create -n fsl-eac python=3.12
conda activate fsl-eac
```

### Step 3: Install PyTorch

```bash
# CUDA 12.8
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

```

### Step 4: Install SSAMBA Dependencies

```bash
cd FSL-EAC/ssamba
git clone https://github.com/SiavashShams/ssamba.git
```

### Step 5: Install PANNS Dependencies

```bash
cd FSL-EAC/PANN
git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git
```

### Step 6: Install Additional Dependencies

```bash
cd FSL-EAC
pip install -r requirements.txt
pip install librosa==0.9.2
pip install tqdm
pip install numpy
```

### Step 7: Download Pretrained Models

**SSAMBA Pretrained Model**:
```bash
https://drive.google.com/drive/u/1/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX
```

**PANNS Pretrained Model**:
```bash
https://zenodo.org/records/3987831/files/Cnn14_16k_mAP=0.438.pth?download=1
```

---

## 📂 Dataset Preparation

### Download DCASE2018 Task 5 Dataset (About 40GB)

```bash
https://zenodo.org/records/1247102
```

### Dataset Structure

```
dataset/DCASE2018/DCASE2018-task5-dev/
├── audio/
│   ├── train/
│   └── test/
├── meta.txt
└── evaluation_setup/
```

The `meta.txt` file should contain:
```
audio/train/a001_10_20.wav	absence	a001
audio/train/a001_30_40.wav	absence	a001
...
```

---

## 🧪 Experiments

### Experiment 1: SSAMBA Transductive FT (Baseline)

**Objective**: Establish baseline with SSAMBA + Transductive FT

```bash
python ssamba_transductive.py \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30 \
  --save_path ssamba_baseline.pth
```

**Result**: 75.87%

---

### Experiment 2: No Augmentation (Ablation)

**Objective**: Measure impact of data augmentation

```bash
python ssamba_transductive_noaug.py \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30
```

**Result**: 68.09% (-7.78% vs baseline, proves augmentation critical)

---

### Experiment 3: Prototypical Networks

**Objective**: Compare with traditional meta-learning approach

```bash
python baseline_prototypical.py \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30
```

**Result**: 64.00% (-11.87% vs Transductive FT)

---

### Experiment 4: PANNS Baseline

**Objective**: Test CNN architecture with native 2048D features

```bash
python panns_transductive.py \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30 \
  --save_path panns_baseline.pth
```

**Result**: 87.02% (**Best performance**)

---

### Experiment 5: PANNS 768D (Dimension Ablation)

**Objective**: Isolate dimension vs architecture effect

```bash
python panns_transductive_768d.py \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30
```

**Result**: 86.58% (only -0.44% drop, architecture matters more than dimension)

---

### Experiment 6: SSAMBA Matched Parameters

**Objective**: Fair comparison at similar trainable parameters (~44M)

```bash
python ssamba_transductive.py \
  --freeze_layers 13 \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30 \
  --save_path ssamba_matched.pth
```

**Result**: 85.47% (still -1.55% below PANNS)

---

### Experiment 7: SSAMBA 2048D Multi-Layer Fusion

**Objective**: Enhance SSAMBA with multi-layer features (L8+16+24 → 2048D)

```bash
python ssamba_transductive_2048d.py \
  --freeze_layers 16 \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30 \
  --save_path ssamba_2048d.pth
```

**Result**: 82.18% (+6.31% over 768D baseline, but still -4.84% below PANNS)

---

## 🔁 Quick Reproduction

### Run All Experiments

```bash
# 1. SSAMBA Baseline
nohup python -u ssamba_transductive.py > logs/train_baseline.log 2>&1 &

# 2. No Augmentation
nohup python -u ssamba_transductive_noaug.py > logs/train_noaug.log 2>&1 &

# 3. Prototypical Networks
nohup python -u baseline_prototypical.py > logs/train_prototypical.log 2>&1 &

# 4. PANNS Baseline
nohup python -u panns_transductive.py > logs/train_panns.log 2>&1 &

# 5. PANNS 768D
nohup python -u panns_transductive_768d.py > logs/train_panns_768d.log 2>&1 &

# 6. SSAMBA Matched
nohup python -u ssamba_transductive.py --freeze_layers 13 > logs/train_ssamba_matched.log 2>&1 &

# 7. SSAMBA 2048D
nohup python -u ssamba_transductive_2048d.py --freeze_layers 16 > logs/train_ssamba_2048d.log 2>&1 &
```

### Monitor Training

```bash
tail -f logs/train_baseline.log
```

---

## 📁 Project Structure

```
FSL-EAC/
├── ssamba_transductive.py          # SSAMBA baseline (768D)
├── ssamba_transductive_noaug.py    # No augmentation ablation
├── ssamba_transductive_2048d.py    # Multi-layer fusion (2048D)
├── baseline_prototypical.py        # Prototypical Networks
├── panns_transductive.py           # PANNS baseline (2048D)
├── panns_transductive_768d.py      # PANNS dimension ablation
├── specaugment.py                  # SpecAugment implementation
├── TECHNICAL_REPORT.md             # Comprehensive technical report
├── README.md                       # This file
├── ssamba/                         # SSAMBA model code
│   ├── src/
│   ├── Vim/
│   └── ssamba_base_400.pth        # Pretrained weights
├── PANN/                           # PANNS model code
│   └── Cnn14_mAP=0.431.pth        # Pretrained weights
└── dataset/
    └── DCASE2018/
        └── DCASE2018-task5-dev/   # Dataset
```

---

## 🎓 Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | AdamW optimizer |
| Fine-tune Steps | 5 | Per episode |
| Entropy Weight (λ) | 0.1 | For transductive loss |
| Augmentation Factor | 4 | SpecAugment multiplier |
| Batch Size | 25-36 | For gradient accumulation |
| N-way K-shot | 5-way 5-shot | Episode configuration |
| Query per class | 15 | Episode configuration |

---

## 📊 Results Summary

### Fine-Tuning Strategy Comparison

| Method | Accuracy | vs Prototypical |
|--------|----------|-----------------|
| Prototypical Networks | 64.00% | Baseline |
| **Transductive FT** | **75.87%** | **+11.87%** ✅ |

### Architecture Comparison (Matched Parameters ~42-44M)

| Model | Architecture | Dimension | Accuracy |
|-------|-------------|-----------|----------|
| SSAMBA | Mamba | 768D | 85.47% |
| **PANNS** | **CNN** | **2048D** | **87.02%** |

### SSAMBA Dimension Enhancement

| Configuration | Feature Extraction | Dimension | Accuracy | Improvement |
|--------------|-------------------|-----------|----------|-------------|
| Baseline | Layer 24 only | 768D | 75.87% | — |
| Multi-Layer | Layers 8+16+24 | 2048D | 82.18% | **+6.31%** |

---



---


