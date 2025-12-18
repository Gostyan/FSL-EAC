# FSL-EAC: 少样本环境音频分类


这是一个关于环境音频分类的少样本学习方法的综合性研究，基于DCASE2018 Task 5数据集，对比了Transductive Fine-Tuning与Prototypical Networks方法，并评估了SSAMBA (Mamba)与PANNS (CNN)架构的性能。

## 🎯 核心结果

| 方法 | 模型 | 特征维度 | 准确率 |
|--------|-------|-------------|----------|
| **Transductive FT** | SSAMBA | 768D | **75.87%** |
| **PANNS (最佳)** | CNN | 2048D | **87.02%** |
| SSAMBA 2048D | Mamba | 2048D | **82.18%** |
| Prototypical | SSAMBA | 768D | 64.00% |

**关键发现**：
- ✅ TransductiveFT比Prototypical Networks提升了**+11.87%**
- ✅ 数据增强(SpecAugment 4×)带来**+7.78%**的提升
- ✅ PANNS (CNN)达到**87.02%**，比SSAMBA (Mamba)高约1-5%
- ✅ 多层特征融合将SSAMBA从75.87%提升至82.18%

---

## 📋 目录

- [概述](#概述)
- [安装](#安装)
- [数据集准备](#数据集准备)
- [实验](#实验)
- [快速复现](#快速复现)
- [项目结构](#项目结构)

---

## 🔍 概述

### 研究问题

本项目研究以下问题：
1. **微调策略**: Transductive FT是否优于Prototypical Networks？
2. **数据增强**: SpecAugment在少样本场景下的效果如何？
3. **架构对比**: SSAMBA (Mamba) vs PANNS (CNN)哪个更适合音频频谱？
4. **维度消融**: 更高的维度(2048D vs 768D)是否有益？
5. **多层特征**: 层融合能否提升SSAMBA性能？

### 方法: Transductive Fine-Tuning

与传统元学习方法不同，我们的方法同时利用支持集和查询集：

```
每个episode：
  1. 从支持集样本提取特征（使用SpecAugment 4×增强）
  2. 用原型初始化分类器
  3. 微调5步：
     - 支持集CE损失：标准分类损失
     - 查询集熵损失：鼓励高置信度预测（无标签）
  4. 在查询集上评估
```

**核心创新**：查询集熵正则化从无标签数据分布中提供额外的学习信号。

---

## 🛠️ 安装

### 环境要求

- Python 3.12
- CUDA 12.8


### 步骤1: 克隆仓库

```bash
git clone https://github.com/Gostyan/FSL-EAC.git
cd FSL-EAC
```

### 步骤2: 创建Conda环境

```bash
conda create -n fsl-eac python=3.12
conda activate fsl-eac
```

### 步骤3: 安装PyTorch

```bash
# CUDA 12.8
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

```

### 步骤4: 安装SSAMBA依赖

```bash
cd FSL-EAC/ssamba
git clone https://github.com/SiavashShams/ssamba.git
```

### 步骤5: 安装PANNS依赖

```bash
cd FSL-EAC/PANN
git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git
```

### 步骤6: 安装其他依赖

```bash
cd FSL-EAC
pip install -r requirements.txt
pip install librosa==0.9.2
pip install tqdm
pip install numpy
```

### 步骤7: 下载预训练模型

**SSAMBA预训练模型**：
```bash
https://drive.google.com/drive/u/1/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX
```

**PANNS预训练模型**：
```bash
https://zenodo.org/records/3987831/files/Cnn14_16k_mAP=0.438.pth?download=1
```

---

## 📂 数据集准备

### 下载DCASE2018 Task 5数据集 (约40GB)

```bash
https://zenodo.org/records/1247102
```

### 数据集结构

```
dataset/DCASE2018/DCASE2018-task5-dev/
├── audio/
│   ├── train/
│   └── test/
├── meta.txt
└── evaluation_setup/
```

`meta.txt` 文件应包含：
```
audio/train/a001_10_20.wav	absence	a001
audio/train/a001_30_40.wav	absence	a001
...
```

---

## 🧪 实验

### 实验1: SSAMBA Transductive FT (基线)

**目标**: 建立SSAMBA + Transductive FT基线

```bash
python ssamba_transductive.py \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30 \
  --save_path ssamba_baseline.pth
```

**结果**: 75.87%

---

### 实验2: 无数据增强 (消融实验)

**目标**: 测量数据增强的影响

```bash
python ssamba_transductive_noaug.py \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30
```

**结果**: 68.09% (相比基线-7.78%，证明增强的重要性)

---

### 实验3: Prototypical Networks

**目标**: 与传统元学习方法对比

```bash
python baseline_prototypical.py \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30
```

**结果**: 64.00% (相比Transductive FT低11.87%)

---

### 实验4: PANNS基线

**目标**: 测试CNN架构与原生2048D特征

```bash
python panns_transductive.py \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30 \
  --save_path panns_baseline.pth
```

**结果**: 87.02% (**最佳性能**)

---

### 实验5: PANNS 768D (维度消融)

**目标**: 分离维度与架构的影响

```bash
python panns_transductive_768d.py \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30
```

**结果**: 86.58% (仅下降0.44%，架构比维度更重要)

---

### 实验6: SSAMBA匹配参数

**目标**: 在相似可训练参数(~44M)下公平对比

```bash
python ssamba_transductive.py \
  --freeze_layers 13 \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30 \
  --save_path ssamba_matched.pth
```

**结果**: 85.47% (仍比PANNS低1.55%)

---

### 实验7: SSAMBA 2048D多层融合

**目标**: 用多层特征增强SSAMBA (L8+16+24 → 2048D)

```bash
python ssamba_transductive_2048d.py \
  --freeze_layers 16 \
  --num_epochs 50 \
  --episodes_per_epoch 50 \
  --eval_episodes 30 \
  --save_path ssamba_2048d.pth
```

**结果**: 82.18% (相比768D基线+6.31%，但仍比PANNS低4.84%)

---

## 🔁 快速复现

### 运行所有实验

```bash
# 1. SSAMBA基线
nohup python -u ssamba_transductive.py > logs/train_baseline.log 2>&1 &

# 2. 无数据增强
nohup python -u ssamba_transductive_noaug.py > logs/train_noaug.log 2>&1 &

# 3. Prototypical Networks
nohup python -u baseline_prototypical.py > logs/train_prototypical.log 2>&1 &

# 4. PANNS基线
nohup python -u panns_transductive.py > logs/train_panns.log 2>&1 &

# 5. PANNS 768D
nohup python -u panns_transductive_768d.py > logs/train_panns_768d.log 2>&1 &

# 6. SSAMBA匹配参数
nohup python -u ssamba_transductive.py --freeze_layers 13 > logs/train_ssamba_matched.log 2>&1 &

# 7. SSAMBA 2048D
nohup python -u ssamba_transductive_2048d.py --freeze_layers 16 > logs/train_ssamba_2048d.log 2>&1 &
```

### 监控训练

```bash
tail -f logs/train_baseline.log
```

---

## 📁 项目结构

```
FSL-EAC/
├── ssamba_transductive.py          # SSAMBA基线 (768D)
├── ssamba_transductive_noaug.py    # 无增强消融实验
├── ssamba_transductive_2048d.py    # 多层融合 (2048D)
├── baseline_prototypical.py        # Prototypical Networks
├── panns_transductive.py           # PANNS基线 (2048D)
├── panns_transductive_768d.py      # PANNS维度消融
├── specaugment.py                  # SpecAugment实现
├── TECHNICAL_REPORT.md             # 完整技术报告
├── README.md                       # 英文说明文档
├── README_CN.md                    # 中文说明文档（本文件）
├── ssamba/                         # SSAMBA模型代码
│   ├── src/
│   ├── Vim/
│   └── ssamba_base_400.pth        # 预训练权重
├── PANN/                           # PANNS模型代码
│   └── Cnn14_mAP=0.431.pth        # 预训练权重
└── dataset/
    └── DCASE2018/
        └── DCASE2018-task5-dev/   # 数据集
```

---

## 🎓 关键超参数

| 参数 | 值 | 说明 |
|-----------|-------|-------------|
| 学习率 | 1e-4 | AdamW优化器 |
| 微调步数 | 5 | 每个episode |
| 熵权重 (λ) | 0.1 | 用于transductive损失 |
| 增强因子 | 4 | SpecAugment倍数 |
| 批大小 | 25-36 | 用于梯度累积 |
| N-way K-shot | 5-way 5-shot | Episode配置 |
| 每类查询样本数 | 15 | Episode配置 |

---

## 📊 结果汇总

### 微调策略对比

| 方法 | 准确率 | vs Prototypical |
|--------|----------|-----------------|
| Prototypical Networks | 64.00% | 基线 |
| **Transductive FT** | **75.87%** | **+11.87%** ✅ |

### 架构对比 (匹配参数 ~42-44M)

| 模型 | 架构 | 维度 | 准确率 |
|-------|-------------|-----------|----------|
| SSAMBA | Mamba | 768D | 85.47% |
| **PANNS** | **CNN** | **2048D** | **87.02%** |

### SSAMBA维度增强

| 配置 | 特征提取 | 维度 | 准确率 | 提升 |
|--------------|-------------------|-----------|----------|-------------|
| 基线 | 仅Layer 24 | 768D | 75.87% | — |
| 多层融合 | Layers 8+16+24 | 2048D | 82.18% | **+6.31%** |

---



---


