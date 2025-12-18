#!/usr/bin/env python3
"""
PANNS Transductive Fine-Tuning - Native 768D Version

消融实验: 测试PANNS的优势是否来自更高的特征维度
- PANNS原始架构: conv_blocks → fc1(2048D) → 分类层
- 本版本: conv_blocks → feature_projection(768D) → transductive分类器
- 移除fc1分类头，类似SSAMBA的做法
- 直接输出768D原生特征，而非2048D压缩

目的: 如果768D的PANNS性能仍优于SSAMBA，说明优势来自CNN架构而非维度
"""

import os
import sys
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import librosa

# 添加PANNS路径
sys.path.insert(0, "/root/bypass")
sys.path.insert(0, "/root/bypass/PANN/audioset_tagging_cnn/pytorch")

from specaugment import get_specaugment_light


# ============================================================================
#  数据加载 (使用64 mel bins for PANNS)
# ============================================================================

class AudioItem:
    def __init__(self, path, label, session):
        self.path = path
        self.label = label
        self.session = session


def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def extract_log_mel_spectrogram(y, sr=16000, n_mels=64, n_fft=1024, 
                                 hop_length=160, win_length=400):
    """提取64-bin LogMel频谱图 (PANNS配置)"""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-9)
    return S_db


class MetaLearningDataset:
    """支持训练/测试划分的 Meta-Learning 数据集"""
    
    def __init__(self, root, n_way=5, k_shot=5, q_query=15, sr=16000, n_mels=64,
                 train_ratio=0.7, seed=42, verbose=True):
        self.root = root
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.sr = sr
        self.n_mels = n_mels
        
        random.seed(seed)
        np.random.seed(seed)
        
        meta_path = os.path.join(root, "meta.txt")
        self.items = self._parse_meta_file(meta_path)
        
        self.by_label = defaultdict(list)
        self.sessions_by_label = defaultdict(set)
        
        for item in self.items:
            self.by_label[item.label].append(item)
            self.sessions_by_label[item.label].add(item.session)
        
        self.labels = list(self.by_label.keys())
        
        self.train_sessions = {}
        self.test_sessions = {}
        self._split_sessions(train_ratio)
        
        self.train_items = defaultdict(list)
        self.test_items = defaultdict(list)
        
        for label in self.labels:
            for item in self.by_label[label]:
                if item.session in self.train_sessions[label]:
                    self.train_items[label].append(item)
                else:
                    self.test_items[label].append(item)
        
        if verbose:
            self._print_split_info()
    
    def _parse_meta_file(self, meta_path):
        items = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    rel_path, label, session = parts[0], parts[1], parts[2]
                    full_path = os.path.join(os.path.dirname(meta_path), rel_path)
                    if os.path.exists(full_path):
                        items.append(AudioItem(full_path, label, session))
        return items
    
    def _split_sessions(self, train_ratio):
        for label in self.labels:
            sessions = sorted(self.sessions_by_label[label])
            random.shuffle(sessions)
            n_train = max(2, int(len(sessions) * train_ratio))
            n_train = min(n_train, len(sessions) - 2)
            self.train_sessions[label] = set(sessions[:n_train])
            self.test_sessions[label] = set(sessions[n_train:])
    
    def _print_split_info(self):
        print(f"\n[MetaLearningDataset] {len(self.items)} samples, {len(self.labels)} classes")
        print(f"[Config] Using {self.n_mels} mel bins (PANNS configuration)")
        print("-" * 60)
        for label in sorted(self.labels):
            n_train = len(self.train_items[label])
            n_test = len(self.test_items[label])
            n_train_sess = len(self.train_sessions[label])
            n_test_sess = len(self.test_sessions[label])
            print(f"  {label:20s}: Train={n_train:5d} ({n_train_sess:2d} sess) | Test={n_test:5d} ({n_test_sess:2d} sess)")
        print("-" * 60)
    
    def _load_spectrogram(self, item):
        y = load_audio(item.path, sr=self.sr)
        spec = extract_log_mel_spectrogram(y, sr=self.sr, n_mels=self.n_mels)
        return torch.from_numpy(spec).float()
    
    def _pad_spectrogram(self, spec, target_len=1024):
        _, T = spec.shape
        if T < target_len:
            pad = torch.zeros(spec.shape[0], target_len - T)
            spec = torch.cat([spec, pad], dim=1)
        else:
            spec = spec[:, :target_len]
        return spec
    
    def sample_episode(self, split="train", target_len=1024):
        items_dict = self.train_items if split == "train" else self.test_items
        sessions_dict = self.train_sessions if split == "train" else self.test_sessions
        
        valid_classes = [lab for lab in self.labels if len(sessions_dict[lab]) >= 2]
        selected_classes = random.sample(valid_classes, min(self.n_way, len(valid_classes)))
        
        support_specs, support_labels = [], []
        query_specs, query_labels = [], []
        
        for class_idx, label in enumerate(selected_classes):
            sessions = list(sessions_dict[label])
            random.shuffle(sessions)
            
            items_by_session = defaultdict(list)
            for item in items_dict[label]:
                items_by_session[item.session].append(item)
            
            mid = max(1, len(sessions) // 2)
            support_sessions = sessions[:mid]
            query_sessions = sessions[mid:] if mid < len(sessions) else sessions[:1]
            
            # Support samples
            support_pool = []
            for sess in support_sessions:
                support_pool.extend(items_by_session[sess])
            support_samples = random.sample(support_pool, min(self.k_shot, len(support_pool)))
            
            for item in support_samples:
                spec = self._load_spectrogram(item)
                spec = self._pad_spectrogram(spec, target_len)
                support_specs.append(spec)
                support_labels.append(class_idx)
            
            # Query samples
            query_pool = []
            for sess in query_sessions:
                query_pool.extend(items_by_session[sess])
            query_samples = random.sample(query_pool, min(self.q_query, len(query_pool)))
            
            for item in query_samples:
                spec = self._load_spectrogram(item)
                spec = self._pad_spectrogram(spec, target_len)
                query_specs.append(spec)
                query_labels.append(class_idx)
        
        return (torch.stack(support_specs), torch.tensor(support_labels, dtype=torch.long),
                torch.stack(query_specs), torch.tensor(query_labels, dtype=torch.long))


# ============================================================================
#  PANNS 编码器
# ============================================================================

class PANNSEncoder(nn.Module):
    """PANNS Cnn14 编码器 - 原生768维输出(移除fc1分类头)"""
    
    def __init__(self, pretrained_path, input_tdim=1024, freeze_early_blocks=5):
        super().__init__()
        self.conv_output_dim = 2048  # conv_block6输出维度
        self.embed_dim = 768   # 最终输出维度(匹配SSAMBA)
        self.input_tdim = input_tdim
        self.n_mels = 64
        
        self._load_pretrained_model(pretrained_path)
        self._freeze_early_blocks(freeze_early_blocks)
        self._print_params()
    
    def _load_pretrained_model(self, pretrained_path):
        """加载PANNS Cnn14预训练模型"""
        from models import Cnn14_no_specaug
        
        # PANNS Cnn14 for 16kHz
        sample_rate = 16000
        window_size = 512
        hop_size = 160
        mel_bins = 64
        fmin = 50
        fmax = 8000
        classes_num = 527  # AudioSet classes
        
        # 创建模型
        model = Cnn14_no_specaug(
            sample_rate=sample_rate, 
            window_size=window_size,
            hop_size=hop_size, 
            mel_bins=mel_bins, 
            fmin=fmin, 
            fmax=fmax,
            classes_num=classes_num
        )
        
        # 加载预训练权重
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        
        # PANNS checkpoint格式: {'iteration': ..., 'model': state_dict, 'sampler': ...}
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 移除module前缀（如果有DataParallel包装）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        
        # 提取需要的组件（跳过spectrogram/logmel提取器和fc1分类头）
        self.bn0 = model.bn0
        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3
        self.conv_block4 = model.conv_block4
        self.conv_block5 = model.conv_block5
        self.conv_block6 = model.conv_block6
        # 不加载fc1 - 类似SSAMBA去掉分类头的做法
        
        # 添加新的768D特征层（随机初始化，不使用预训练权重）
        self.feature_projection = nn.Linear(self.conv_output_dim, self.embed_dim)
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.zeros_(self.feature_projection.bias)
        
        print(f"[PANNSEncoder] Model loaded from {pretrained_path}")
        print(f"[PANNSEncoder] Removed fc1 classification head (like SSAMBA)")
        print(f"[PANNSEncoder] Added new feature projection: {self.conv_output_dim}D -> {self.embed_dim}D")
        print(f"[PANNSEncoder] Input: {self.n_mels} mel bins")
    
    def _freeze_early_blocks(self, n_blocks):
        """
        冻结策略 - 平衡SSAMBA比例和模型适应能力
        
        PANNS参数分布:
        - conv_block1-5: 18.85M (23.6%)
        - conv_block6.conv1+bn1: ~28M (35%)
        - conv_block6.conv2+bn2: ~28M (35%)
        - fc1: 4.20M (5.3%)
        
        测试结果:
        - 保留conv_block6全部: 71.1%可训练 (太高)
        - 保留conv2+bn2: ~35%可训练 (折中方案)
        - 仅保留bn2: 0.0%可训练 (太低，无法适应)
        
        最终策略: 保留conv_block6.conv2+bn2可训练
        - 可训练参数: ~35% (虽高于SSAMBA的17%，但考虑到PANNS参数集中在最后一层，这是合理的折中)
        - 说明: PANNS的参数分布与SSAMBA不同，很难达到完全相同的比例
        """
        # 冻结所有早期层
        blocks_to_freeze = [
            self.bn0, self.conv_block1, self.conv_block2,
            self.conv_block3, self.conv_block4, self.conv_block5
        ]
        
        for block in blocks_to_freeze:
            for param in block.parameters():
                param.requires_grad = False
        
        # 部分冻结conv_block6: 冻结conv1+bn1，保留conv2+bn2
        for param in self.conv_block6.conv1.parameters():
            param.requires_grad = False
        for param in self.conv_block6.bn1.parameters():
            param.requires_grad = False
        # conv_block6.conv2 和 bn2 保持可训练
        # feature_projection 保持可训练（新添加的层）
        
        print(f"[PANNSEncoder] Frozen: bn0 + conv_block1-5 + conv_block6.conv1/bn1")
        print(f"[PANNSEncoder] Trainable: conv_block6.conv2/bn2 + feature_projection(768D)")

    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100 * trainable / total if total > 0 else 0
        print(f"[PANNSEncoder] Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M ({pct:.1f}%)")
    
    def forward(self, x):
        """
        输入: (batch, n_mels=64, time_steps)
        输出: (batch, 2048) embedding
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, n_mels=64, time_steps=1024)
        
        # 确保输入是64 mel bins
        if x.shape[2] != self.n_mels:
            raise ValueError(f"Expected {self.n_mels} mel bins, got {x.shape[2]}")
        
        # PANNS forward pass (跳过spectrogram/logmel提取)
        # 原始PANNS: (batch, 1, time, freq) -> transpose(1,3) -> (batch, freq, time, 1)
        # 我们的输入: (batch, 1, freq=64, time) -> transpose(1,2) -> (batch, freq, 1, time)
        # 然后transpose(2,3) -> (batch, freq, time, 1) 以匹配PANNS
        x = x.permute(0, 2, 3, 1)  # (batch, mels=64, time, 1)
        x = self.bn0(x)  # bn0 is BatchNorm2d(64)
        x = x.permute(0, 3, 1, 2)  # (batch, 1, mels, time)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2  # (batch, 2048) - conv_block6的pooled输出
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 直接投影到768维（无fc1，无ReLU，类似SSAMBA的做法）
        embedding = self.feature_projection(x)  # (batch, 768)
        
        return embedding


# ============================================================================
#  Transductive 分类器和训练器
# ============================================================================

class TransductiveClassifier(nn.Module):
    """Transductive分类器 - W由原型初始化，b可学习"""
    
    def __init__(self, embed_dim, n_way, temperature=10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_way = n_way
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.W = nn.Parameter(torch.zeros(n_way, embed_dim))
        self.b = nn.Parameter(torch.zeros(n_way))
    
    def init_from_prototypes(self, prototypes):
        with torch.no_grad():
            self.W.data = F.normalize(prototypes, dim=-1)
            self.b.data.zero_()
    
    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        W_norm = F.normalize(self.W, dim=-1)
        logits = torch.mm(x_norm, W_norm.t()) * self.temperature + self.b
        return logits


class TransductiveTrainer:
    """Transductive Fine-Tuning训练器"""
    
    def __init__(self, encoder, dataset, device="cuda", 
                 lr=1e-4, entropy_weight=0.1, finetune_steps=5,
                 augmenter=None, aug_factor=4, batch_size=25):
        self.encoder = encoder.to(device)
        self.dataset = dataset
        self.device = device
        self.lr = lr
        self.entropy_weight = entropy_weight
        self.finetune_steps = finetune_steps
        self.augmenter = augmenter
        self.aug_factor = aug_factor
        self.batch_size = batch_size
        
        self.encoder_optimizer = AdamW(
            [p for p in encoder.parameters() if p.requires_grad], lr=lr
        )
    
    def compute_entropy_loss(self, logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy
    
    def train_episode(self, support_x, support_y, query_x, query_y):
        self.encoder.train()
        n_way = self.dataset.n_way
        
        # 1. 支持集数据增强扩充
        if self.augmenter and self.aug_factor > 1:
            aug_support_list = [support_x]
            aug_labels_list = [support_y]
            for _ in range(self.aug_factor - 1):
                aug_support_list.append(self.augmenter(support_x.clone()))
                aug_labels_list.append(support_y.clone())
            support_x_aug = torch.cat(aug_support_list, dim=0)
            support_y_aug = torch.cat(aug_labels_list, dim=0)
        else:
            support_x_aug = support_x
            support_y_aug = support_y
        
        # 2. 计算初始原型 (分批处理)
        with torch.no_grad():
            support_feat_list = []
            for i in range(0, support_x_aug.size(0), self.batch_size):
                batch_x = support_x_aug[i:i+self.batch_size]
                batch_feat = self.encoder(batch_x)
                support_feat_list.append(batch_feat)
            support_feat = torch.cat(support_feat_list, dim=0)
            prototypes = torch.stack([
                support_feat[support_y_aug == c].mean(0) for c in range(n_way)
            ])
        
        # 3. 初始化Transductive分类器
        classifier = TransductiveClassifier(
            self.encoder.embed_dim, n_way
        ).to(self.device)
        classifier.init_from_prototypes(prototypes)
        classifier_optimizer = AdamW(classifier.parameters(), lr=self.lr * 10)
        
        # 4. Transductive Fine-Tuning循环 (梯度累积)
        total_loss = 0.0
        ce_loss = 0.0
        entropy_loss = 0.0
        for step in range(self.finetune_steps):
            self.encoder_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            
            # 支持集CE损失 (梯度累积)
            n_support_batches = (support_x_aug.size(0) + self.batch_size - 1) // self.batch_size
            ce_loss_step = 0.0
            for i in range(0, support_x_aug.size(0), self.batch_size):
                batch_x = support_x_aug[i:i+self.batch_size]
                batch_y = support_y_aug[i:i+self.batch_size]
                batch_feat = self.encoder(batch_x)
                batch_logits = classifier(batch_feat)
                batch_ce = F.cross_entropy(batch_logits, batch_y) / n_support_batches
                batch_ce.backward()
                ce_loss_step += batch_ce.item()
            
            # Query集熵损失 (梯度累积)
            n_query_batches = (query_x.size(0) + self.batch_size - 1) // self.batch_size
            entropy_loss_step = 0.0
            for i in range(0, query_x.size(0), self.batch_size):
                batch_x = query_x[i:i+self.batch_size]
                batch_feat = self.encoder(batch_x)
                batch_logits = classifier(batch_feat)
                batch_entropy = self.compute_entropy_loss(batch_logits) * self.entropy_weight / n_query_batches
                batch_entropy.backward()
                entropy_loss_step += batch_entropy.item()
            
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            
            self.encoder_optimizer.step()
            classifier_optimizer.step()
            
            ce_loss = ce_loss_step
            entropy_loss = entropy_loss_step / self.entropy_weight if self.entropy_weight > 0 else 0
            total_loss += ce_loss + entropy_loss_step

        # 5. 最终评估 (分批处理)
        self.encoder.eval()
        with torch.no_grad():
            query_logits_list = []
            for i in range(0, query_x.size(0), self.batch_size):
                batch_x = query_x[i:i+self.batch_size]
                batch_feat = self.encoder(batch_x)
                batch_logits = classifier(batch_feat)
                query_logits_list.append(batch_logits)
            query_logits = torch.cat(query_logits_list, dim=0)
            preds = query_logits.argmax(-1)
            acc = (preds == query_y).float().mean().item()
        
        return {
            "loss": total_loss / self.finetune_steps,
            "acc": acc,
            "ce_loss": ce_loss,
            "entropy_loss": entropy_loss
        }
    
    def train_epoch(self, num_episodes):
        metrics = {"loss": [], "acc": [], "ce_loss": [], "entropy_loss": []}
        
        for _ in tqdm(range(num_episodes), desc="Training", ncols=80):
            support_x, support_y, query_x, query_y = self.dataset.sample_episode("train")
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            m = self.train_episode(support_x, support_y, query_x, query_y)
            for k, v in m.items():
                metrics[k].append(v)
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    @torch.no_grad()
    def evaluate(self, num_episodes):
        self.encoder.eval()
        accs = []
        
        for _ in tqdm(range(num_episodes), desc="Evaluating", ncols=80):
            support_x, support_y, query_x, query_y = self.dataset.sample_episode("test")
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            n_way = self.dataset.n_way
            
            # 支持集增强
            if self.augmenter and self.aug_factor > 1:
                aug_support_list = [support_x]
                aug_labels_list = [support_y]
                for _ in range(self.aug_factor - 1):
                    aug_support_list.append(self.augmenter(support_x.clone()))
                    aug_labels_list.append(support_y.clone())
                support_x = torch.cat(aug_support_list, dim=0)
                support_y = torch.cat(aug_labels_list, dim=0)
            
            # 计算原型 (分批处理)
            support_feat_list = []
            for i in range(0, support_x.size(0), self.batch_size):
                batch_x = support_x[i:i+self.batch_size]
                batch_feat = self.encoder(batch_x)
                support_feat_list.append(batch_feat)
            support_feat = torch.cat(support_feat_list, dim=0)
            prototypes = torch.stack([
                support_feat[support_y == c].mean(0) for c in range(n_way)
            ])
            proto_norm = F.normalize(prototypes, dim=-1)
            
            # Query分类 (分批处理)
            query_feat_list = []
            for i in range(0, query_x.size(0), self.batch_size):
                batch_x = query_x[i:i+self.batch_size]
                batch_feat = self.encoder(batch_x)
                query_feat_list.append(batch_feat)
            query_feat = torch.cat(query_feat_list, dim=0)
            query_norm = F.normalize(query_feat, dim=-1)
            logits = torch.mm(query_norm, proto_norm.t())
            preds = logits.argmax(-1)
            
            acc = (preds == query_y).float().mean().item()
            accs.append(acc)
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        ci95 = 1.96 * std_acc / np.sqrt(len(accs))
        
        return {"acc_mean": mean_acc, "acc_std": std_acc, "acc_ci95": ci95}
    
    def train(self, num_epochs, episodes_per_epoch, eval_episodes, save_path=None):
        print("\n" + "=" * 70)
        print("  PANNS Transductive Fine-Tuning Training")
        print("=" * 70)
        
        best_acc = 0.0
        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            
            train_m = self.train_epoch(episodes_per_epoch)
            print(f"  Train Loss: {train_m['loss']:.4f} (CE: {train_m['ce_loss']:.4f}, "
                  f"Entropy: {train_m['entropy_loss']:.4f})")
            print(f"  Train Acc: {train_m['acc']*100:.2f}%")
            
            test_m = self.evaluate(eval_episodes)
            print(f"  Test Acc: {test_m['acc_mean']*100:.2f}% +/- {test_m['acc_std']*100:.2f}%")
            
            if test_m['acc_mean'] > best_acc:
                best_acc = test_m['acc_mean']
                if save_path:
                    torch.save({
                        'encoder_state_dict': self.encoder.state_dict(),
                        'best_acc': best_acc
                    }, save_path)
                    print(f"  [Saved] Best model (acc={best_acc*100:.2f}%)")
        
        print(f"\n{'='*70}")
        print(f"  Training Complete! Best Acc: {best_acc*100:.2f}%")
        print(f"{'='*70}")
        
        return best_acc


# ============================================================================
#  主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/root/bypass/dataset/DCASE2018/DCASE2018-task5-dev")
    parser.add_argument("--pretrained_path", default="/root/bypass/PANN/Cnn14_16k_mAP=0.438.pth")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=15)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--episodes_per_epoch", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_blocks", type=int, default=5, help="冻结前N个卷积块 (共6个)")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--entropy_weight", type=float, default=0.1)
    parser.add_argument("--finetune_steps", type=int, default=5)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--aug_factor", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=36, help="Batch size for processing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_path", default="panns_768d_best.pth")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    print("=" * 70)
    print("  PANNS 768D - Dimension-Matched with SSAMBA")
    print("=" * 70)
    print(f"\n[Ablation] PANNS 2048D -> Projection -> 768D (match SSAMBA)")
    print(f"[Purpose] Test if performance gap is due to dimension or architecture")
    print(f"[Baseline] PANNS Cnn14 (64 mel bins) vs SSAMBA (128 mel bins)")
    print(f"[Config] {args.n_way}-way {args.k_shot}-shot | {args.num_epochs} epochs x {args.episodes_per_epoch} episodes")
    print(f"[Config] Entropy weight: {args.entropy_weight}, Finetune steps: {args.finetune_steps}")
    print(f"[Config] Aug factor: {args.aug_factor}, Augment: {not args.no_augment}")
    print(f"[Config] Freeze first {args.freeze_blocks}/6 conv blocks")
    
    dataset = MetaLearningDataset(
        args.data_root, args.n_way, args.k_shot, args.q_query,
        train_ratio=args.train_ratio, seed=args.seed, n_mels=64
    )
    
    encoder = PANNSEncoder(args.pretrained_path, freeze_early_blocks=args.freeze_blocks)
    
    augmenter = None if args.no_augment else get_specaugment_light()
    
    trainer = TransductiveTrainer(
        encoder, dataset, device,
        lr=args.lr,
        entropy_weight=args.entropy_weight,
        finetune_steps=args.finetune_steps,
        augmenter=augmenter,
        aug_factor=args.aug_factor,
        batch_size=args.batch_size
    )
    
    print("\n[Initial Evaluation]")
    init_m = trainer.evaluate(args.eval_episodes)
    print(f"  Initial Acc: {init_m['acc_mean']*100:.2f}% +/- {init_m['acc_std']*100:.2f}%")
    
    best_acc = trainer.train(
        args.num_epochs, args.episodes_per_epoch, args.eval_episodes, args.save_path
    )
    
    print(f"\n[Summary] {init_m['acc_mean']*100:.2f}% -> {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
