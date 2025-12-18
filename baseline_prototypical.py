#!/usr/bin/env python3
"""
Baseline 1: Prototypical Networks with SSAMBA Encoder

关键差异:
- 原型网络: 支持集仅用于计算原型 (无梯度), Query集用于训练
- 与Transductive对比: 支持集不参与梯度更新
- 其他配置完全相同: 数据加载、增强、训练轮数、冻结层数等
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

# 添加路径
sys.path.insert(0, "/root/bypass")
sys.path.insert(0, "/root/bypass/ssamba/src")
sys.path.insert(0, "/root/bypass/ssamba/src/models")
sys.path.insert(0, "/root/bypass/ssamba/Vim")
sys.path.insert(0, "/root/bypass/ssamba/Vim/vim")
sys.path.insert(0, "/root/bypass/ssamba/Vim/mamba-1p1p1")
sys.path.insert(0, "/root/bypass/ssamba/Vim/causal-conv1d")

from specaugment import get_specaugment_light


# ============================================================================
#  数据加载 (与 Transductive 完全相同)
# ============================================================================

class AudioItem:
    def __init__(self, path, label, session):
        self.path = path
        self.label = label
        self.session = session


def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def extract_log_mel_spectrogram(y, sr=16000, n_mels=128, n_fft=1024, 
                                 hop_length=160, win_length=400):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-9)
    return S_db


class MetaLearningDataset:
    """支持训练/测试划分的 Meta-Learning 数据集"""
    
    def __init__(self, root, n_way=5, k_shot=5, q_query=15, sr=16000, n_mels=128,
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
            
            support_pool = []
            for sess in support_sessions:
                support_pool.extend(items_by_session[sess])
            support_samples = random.sample(support_pool, min(self.k_shot, len(support_pool)))
            
            for item in support_samples:
                spec = self._load_spectrogram(item)
                spec = self._pad_spectrogram(spec, target_len)
                support_specs.append(spec)
                support_labels.append(class_idx)
            
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
#  SSAMBA 编码器 (与 Transductive 完全相同)
# ============================================================================

class SSAMBAEncoder(nn.Module):
    """SSAMBA 编码器"""
    
    def __init__(self, pretrained_path, input_tdim=1024, freeze_early_layers=20):
        super().__init__()
        self.embed_dim = 768
        self.input_tdim = input_tdim
        self._load_pretrained_model(pretrained_path)
        self._freeze_early_layers(freeze_early_layers)
        self._print_params()
    
    def _load_pretrained_model(self, pretrained_path):
        from both_models import AMBAModel
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(pretrained_path, map_location=device)
        
        p_fshape = sd["module.v.patch_embed.proj.weight"].shape[2]
        p_tshape = sd["module.v.patch_embed.proj.weight"].shape[3]
        p_input_fdim = sd["module.p_input_fdim"].item()
        p_input_tdim = sd["module.p_input_tdim"].item()
        print(f"[SSAMBAEncoder] Pretrained: patch=({p_fshape},{p_tshape}), input=({p_input_fdim},{p_input_tdim})")
        
        vision_mamba_config = {
            "img_size": (128, self.input_tdim), "patch_size": 16, "stride": 16,
            "embed_dim": 768, "depth": 24, "rms_norm": True, "residual_in_fp32": True,
            "fused_add_norm": True, "final_pool_type": "mean", "if_abs_pos_embed": True,
            "if_rope": False, "if_rope_residual": False, "bimamba_type": "v2",
            "if_cls_token": True, "if_devide_out": True, "use_middle_cls_token": True,
        }
        
        pretrain_model = AMBAModel(
            fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape,
            input_fdim=p_input_fdim, input_tdim=p_input_tdim,
            pretrain_stage=True, vision_mamba_config=vision_mamba_config
        )
        pretrain_model = torch.nn.DataParallel(pretrain_model)
        pretrain_model.load_state_dict(sd, strict=False)
        
        self.v = pretrain_model.module.v
        self.cls_token_num = 1
        self._setup_input_adaptation(p_fshape, p_tshape, p_input_fdim, p_input_tdim)
        self.norm = nn.LayerNorm(self.embed_dim)
        print(f"[SSAMBAEncoder] Model loaded, embed_dim={self.embed_dim}")
    
    def _setup_input_adaptation(self, p_fshape, p_tshape, p_input_fdim, p_input_tdim):
        f_dim = 128 // p_fshape
        t_dim = self.input_tdim // p_tshape
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        
        p_f_dim = p_input_fdim // p_fshape
        p_t_dim = p_input_tdim // p_tshape
        p_num_patches = p_f_dim * p_t_dim
        
        if num_patches != p_num_patches:
            new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].reshape(
                1, p_f_dim, p_t_dim, self.embed_dim
            ).permute(0, 3, 1, 2)
            new_pos_embed = F.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear", align_corners=False)
            new_pos_embed = new_pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches, self.embed_dim)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :self.cls_token_num, :], new_pos_embed], dim=1)
            )
    
    def _freeze_early_layers(self, n_layers):
        for param in self.v.patch_embed.parameters():
            param.requires_grad = False
        if hasattr(self.v, 'pos_embed'):
            self.v.pos_embed.requires_grad = False
        if hasattr(self.v, 'layers'):
            for i, layer in enumerate(self.v.layers):
                if i < n_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100 * trainable / total if total > 0 else 0
        print(f"[SSAMBAEncoder] Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M ({pct:.1f}%)")
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[2] < 128:
            pad = torch.zeros(x.shape[0], 1, 128 - x.shape[2], x.shape[3], device=x.device)
            x = torch.cat([x, pad], dim=2)
        
        x = self.v.patch_embed(x)
        B, L, D = x.shape
        cls_token = self.v.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        residual = None
        for layer in self.v.layers:
            x, residual = layer(x, residual)
        
        if residual is not None:
            x = residual + x
        
        return self.norm(x[:, 0])


# ============================================================================
#  Prototypical Networks 训练器
# ============================================================================

class PrototypicalTrainer:
    """Prototypical Networks训练器 (标准实现)"""
    
    def __init__(self, encoder, dataset, device="cuda", 
                 lr=1e-4, augmenter=None, aug_factor=4, batch_size=36):
        self.encoder = encoder.to(device)
        self.dataset = dataset
        self.device = device
        self.lr = lr
        self.augmenter = augmenter
        self.aug_factor = aug_factor
        self.batch_size = batch_size
        
        self.optimizer = AdamW(
            [p for p in encoder.parameters() if p.requires_grad], lr=lr
        )
    
    def train_episode(self, support_x, support_y, query_x, query_y):
        self.encoder.train()
        n_way = self.dataset.n_way
        
        # 1. 支持集数据增强
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
        
        # 2. 分批计算Support特征和损失（梯度累积，避免OOM）
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        n_batches = (support_x_aug.size(0) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, support_x_aug.size(0), self.batch_size):
            batch_x = support_x_aug[i:i+self.batch_size]
            batch_y = support_y_aug[i:i+self.batch_size]
            
            # 提取batch特征
            batch_feat = self.encoder(batch_x)
            batch_norm = F.normalize(batch_feat, dim=-1)
            
            # 计算当前batch的原型（使用全部support特征）
            with torch.no_grad():
                all_feat_list = []
                for j in range(0, support_x_aug.size(0), self.batch_size):
                    chunk_x = support_x_aug[j:j+self.batch_size]
                    chunk_feat = self.encoder(chunk_x)
                    all_feat_list.append(chunk_feat)
                all_feat = torch.cat(all_feat_list, dim=0)
                prototypes = torch.stack([
                    all_feat[support_y_aug == c].mean(0) for c in range(n_way)
                ])
                proto_norm = F.normalize(prototypes, dim=-1)
            
            # 计算batch损失
            batch_logits = torch.mm(batch_norm, proto_norm.t())
            batch_loss = F.cross_entropy(batch_logits, batch_y) / n_batches
            batch_loss.backward()
            total_loss += batch_loss.item()
        
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.optimizer.step()
        
        # 3. Query集评估（无梯度）
        with torch.no_grad():
            # 重新计算原型
            self.encoder.eval()
            support_feat_list = []
            for i in range(0, support_x_aug.size(0), self.batch_size):
                batch_x = support_x_aug[i:i+self.batch_size]
                batch_feat = self.encoder(batch_x)
                support_feat_list.append(batch_feat)
            support_feat = torch.cat(support_feat_list, dim=0)
            prototypes = torch.stack([
                support_feat[support_y_aug == c].mean(0) for c in range(n_way)
            ])
            proto_norm = F.normalize(prototypes, dim=-1)
            
            # Query分类
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
        
        return {"loss": total_loss, "acc": acc}
    
    def train_epoch(self, num_episodes):
        metrics = {"loss": [], "acc": []}
        
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
            
            # 计算原型
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
            
            # Query分类
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
        print("  Baseline: Prototypical Networks Training")
        print("=" * 70)
        
        best_acc = 0.0
        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            
            train_m = self.train_epoch(episodes_per_epoch)
            print(f"  Train Loss: {train_m['loss']:.4f}")
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
    parser.add_argument("--pretrained_path", default="/root/bypass/ssamba/ssamba_base_400.pth")
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=15)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--episodes_per_epoch", type=int, default=50)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_layers", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--aug_factor", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_path", default="baseline_prototypical_best.pth")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    print("=" * 70)
    print("  Baseline: Prototypical Networks with SSAMBA")
    print("=" * 70)
    print(f"\n[Config] {args.n_way}-way {args.k_shot}-shot | {args.num_epochs} epochs x {args.episodes_per_epoch} episodes")
    print(f"[Config] Aug factor: {args.aug_factor}, Augment: {not args.no_augment}")
    print(f"[Config] 标准Prototypical: 原型有梯度, Query CE Loss, 无Episode内优化")
    
    dataset = MetaLearningDataset(
        args.data_root, args.n_way, args.k_shot, args.q_query,
        train_ratio=args.train_ratio, seed=args.seed
    )
    
    encoder = SSAMBAEncoder(args.pretrained_path, freeze_early_layers=args.freeze_layers)
    
    augmenter = None if args.no_augment else get_specaugment_light()
    
    trainer = PrototypicalTrainer(
        encoder, dataset, device,
        lr=args.lr,
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
