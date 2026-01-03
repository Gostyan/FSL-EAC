# 算法伪代码

## Algorithm 1: SpecAugment Data Augmentation

```
Algorithm 1: SpecAugment for Audio Spectrograms
Input: Mel-spectrogram X ∈ R^(F×T), where F=frequency bins, T=time frames
       time_mask_param τ_t, freq_mask_param τ_f
Output: Augmented spectrogram X' ∈ R^(F×T)

1: X' ← X  // Copy input spectrogram
2: 
3: // Time Masking
4: t ← Random(0, min(τ_t, T))  // Sample mask width
5: t_0 ← Random(0, T - t)       // Sample mask start position
6: X'[:, t_0:t_0+t] ← 0         // Mask time frames
7: 
8: // Frequency Masking  
9: f ← Random(0, min(τ_f, F))   // Sample mask width
10: f_0 ← Random(0, F - f)      // Sample mask start position
11: X'[f_0:f_0+f, :] ← 0        // Mask frequency bins
12:
13: return X'
```

**Parameters**:
- τ_t = 30 (time mask parameter)
- τ_f = 15 (frequency mask parameter)
- Applied 4× to each support sample

---

## Algorithm 2: Transductive Fine-Tuning for Few-Shot Learning

```
Algorithm 2: Transductive Fine-Tuning
Input: Encoder θ, Support set S = {(x_i^s, y_i^s)}_{i=1}^{N_s}, 
       Query set Q = {x_j^q}_{j=1}^{N_q}, N-way classes
       Hyperparameters: learning rate α, entropy weight λ, finetune steps T
Output: Query predictions ŷ

// Step 1: Data Augmentation
1: S_aug ← ∅
2: for each (x, y) in S do
3:     S_aug ← S_aug ∪ {(x, y)}  // Original sample
4:     for k = 1 to 3 do
5:         x' ← SpecAugment(x, τ_t=30, τ_f=15)
6:         S_aug ← S_aug ∪ {(x', y)}  // Augmented sample
7:     end for
8: end for
// S_aug now contains N_s × 4 samples

// Step 2: Compute Prototypes
9: for c = 1 to N do
10:     f_c ← {f_θ(x) | (x,y) ∈ S_aug, y = c}  // Extract features for class c
11:     p_c ← Mean(f_c)                         // Compute prototype
12: end for
13: P ← {p_1, ..., p_N}

// Step 3: Initialize Classifier
14: W ← Normalize([p_1; ...; p_N])^T  // Weight matrix: N × d
15: b ← 0  // Bias vector: N
16: Create classifier h(x; W, b) = softmax(W^T f_θ(x) / τ + b)
17: τ ← 10.0  // Temperature

// Step 4: Transductive Fine-Tuning
18: for t = 1 to T do  // T = 5 finetune steps
19:     
20:     // Compute Support Set Loss (Supervised)
21:     L_CE ← 0
22:     for each (x, y) in S_aug do
23:         p̂ ← h(x; W, b)  // Forward pass
24:         L_CE ← L_CE - log(p̂_y)
25:     end for
26:     L_CE ← L_CE / |S_aug|
27:     
28:     // Compute Query Set Entropy Loss (Unsupervised)
29:     L_entropy ← 0
30:     for each x in Q do
31:         p̂ ← h(x; W, b)  // Forward pass
32:         L_entropy ← L_entropy - Σ_{c=1}^N p̂_c log(p̂_c)
33:     end for
34:     L_entropy ← L_entropy / |Q|
35:     
36:     // Combined Loss
37:     L_total ← L_CE + λ × L_entropy
38:     
39:     // Gradient Update
40:     θ, W, b ← θ, W, b - α × ∇_{θ,W,b} L_total
41: end for

// Step 5: Predict on Query Set
42: for each x in Q do
43:     ŷ ← argmax h(x; W, b)
44: end for
45: return ŷ
```

**Key Parameters**:
- N = 5 (5-way classification)
- K = 5 (5-shot per class)
- |S| = N × K = 25 (support size before augmentation)
- |S_aug| = 100 (after 4× augmentation)
- |Q| = N × 15 = 75 (query size)
- T = 5 (finetune steps)
- α = 1e-4 (learning rate)
- λ = 0.1 (entropy weight)
- τ = 10.0 (temperature)

---

## Algorithm 3: Meta-Training Loop

```
Algorithm 3: Meta-Training with Transductive Fine-Tuning
Input: Dataset D with train/test split, Encoder θ, Episodes per epoch E
Output: Trained encoder θ*

1: θ ← Load pretrained weights (AudioSet)
2: 
3: for epoch = 1 to MaxEpochs do
4:     
5:     // Training Phase
6:     for episode = 1 to E do
7:         (S, Q) ← SampleEpisode(D_train, N=5, K=5, Q=15)
8:         ŷ ← TransductiveFineTuning(θ, S, Q)  // Algorithm 2
9:         acc ← Accuracy(ŷ, Q.labels)
10:     end for
11:     
12:     // Evaluation Phase
13:     accs ← []
14:     for episode = 1 to 30 do  // 30 test episodes
15:         (S, Q) ← SampleEpisode(D_test, N=5, K=5, Q=15)
16:         ŷ ← TransductiveFineTuning(θ, S, Q)
17:         accs.append(Accuracy(ŷ, Q.labels))
18:     end for
19:     
20:     test_acc ← Mean(accs)
21:     test_std ← StdDev(accs)
22:     
23:     if test_acc > best_acc then
24:         θ* ← θ
25:         best_acc ← test_acc
26:     end if
27:     
28:     Print("Epoch {}: Test Acc: {:.2f}% ± {:.2f}%", epoch, test_acc, test_std)
29: end for
30:
31: return θ*
```

**Training Configuration**:
- MaxEpochs = 50
- E = 50 (episodes per training epoch)
- Evaluation episodes = 30
- Optimizer: AdamW with learning rate 1e-4

---

## Computational Complexity

**Per Episode**:
- Data Augmentation: O(N_s × F × T) for 4× augmentation
- Feature Extraction: O(|S_aug| × C_encoder) + O(|Q| × C_encoder)
  - C_encoder ≈ O(L × d²) for L=24 Mamba layers, d=768 hidden dim
- Prototype Computation: O(|S_aug| × d) 
- Fine-tuning (T steps): O(T × (|S_aug| + |Q|) × d × N)

**Total per epoch**:
- Training: O(E × [augmentation + feature + finetune])
- Evaluation: O(30 × [augmentation + feature + finetune])

**Memory**:
- Peak GPU memory: ~12GB for batch processing
- Gradient accumulation used to handle |S_aug|=100 samples
