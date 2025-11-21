# Transformer-Gated Skip Connection (TGSC) 模型改进说明文档

## 1. 改进动机与总体思路

### 1.1 背景问题 (Pain Point)

在混合型 CNN + Transformer 的结构中（如 TransUNet），常见的瓶颈在于：

- **全局信息仅限于瓶颈**：Transformer 捕获了图像的 **全局语义关系**，但其输出（`memory`）通常只作为解码器最深层的**初始输入**，在后续的上采样过程中被逐层稀释。
- **局部融合的局限性**：解码阶段的跳跃连接（Skip Connection）多为 **简单拼接（`concat`）\**或\**局部门控（`AttentionGate`）**。`AttentionGate` 仅利用 *解码器上一层* 的局部特征作为门控信号，无法利用到全局上下文。
- **结果**：**全局语义与局部细节割裂**。解码器的高分辨率层（如 `f1`, `f0`）虽然恢复了细节，但也重新引入了大量的**无效纹理和背景噪声**（例如，草地上的纹理被误判为边缘），导致最终的边缘检测结果出现伪边缘（false positives）与断裂（discontinuity）。

### 1.2 改进核心 (Our Solution: TGSC)

为解决上述问题，本研究提出 **Transformer-Gated Skip Connection (TGSC)**：

> **核心机制**：我们不再将 Transformer 的全局特征（`global memory`）仅用作一次性的初始输入，而是将其**上采样并广播（broadcast）到解码器的每一层**，让它**直接充当**跳跃连接的**门控信号**，以此指导每个解码层对 CNN 局部特征（`f0`...`f3`）进行自适应的筛选与增强。

**核心思想**： TGSC 实现了一种**“全局-局部”信息的动态协同融合**。它迫使模型在融合高分辨率局部细节（来自CNN）时，必须“参考”全局语义上下文（来自Transformer），从而在解码的每一步都保持语义一致的边缘预测。

## 2. 模型结构与改进策略

### 2.1 总体结构概览

- **原模型 (Baseline)**： ResNet Backbone → Transformer Encoder (at `f4`) → U-Net Decoder + `AttentionGate` (gating signal 来自 *上一层decoder*) → Edge Head
- **改进后 (Ours)**： ResNet Backbone → Transformer Encoder (at `f4`) → **TGSC-Enhanced Decoder** (gating signal 来自 *Transformer* `memory`) → Edge Head

### 2.2 TGSC 模块机制 (Mechanism)

对于解码器的第 $k$ 层（例如，融合 `f_k` 的那一层）：

1. **全局门控信号 (Global Gate)**： 将 Transformer 输出的全局特征 `memory`（形状 `B, C_mem, H_low, W_low`）通过双线性插值上采样，使其空间分辨率与来自 ResNet 的局部特征 `f_k`（形状 `B, C_k, H_k, W_k`）**保持一致**，得到 `M_k`。 `M_k = Upsample(memory, size=f_k.shape[2:])`

2. **门控权重生成 (Attention Coefficient)**： 使用轻量级的 $1 \times 1$ 卷积将 `f_k` 和 `M_k` 投影到同一中间维度（例如 `C_k // 2`），相加后通过 `ReLU` 和 `Sigmoid` 生成空间注意力图谱 $\alpha_k$。

   $W_f = \text{Conv1x1}(f_k)$ $W_m = \text{Conv1x1}(M_k)$ $\alpha_k = \sigma(\text{Conv1x1}(\text{ReLU}(W_f + W_m)))$ （其中 $\sigma$ 是 Sigmoid 函数，$\alpha_k$ 的形状为 `B, 1, H_k, W_k`）

3. **特征调制 (Feature Modulation)**： 利用该注意力图谱 $\alpha_k$ 按元素乘法来调制（筛选）原始的局部特征 `f_k`。 $\hat{f}_k = \alpha_k \odot f_k$

4. **特征融合 (Feature Fusion)**： 将调制后的特征 $\hat{f}_k$（已“过滤”掉噪声）与来自解码器上一层的上采样特征 `x_up` 进行拼接，并送入后续卷积。 $y_k = \text{Conv}(\text{Concat}(x_{up}, \hat{f}_k))$

### 2.3 模块设计建议

- **Conv1×1 投影**：统一通道维度、减少计算；

- **共享 memory 投影层**：对所有 skip 共享参数，保持语义一致性；

- **Sigmoid 门控 + 残差保留**：

  $$\hat{f}_k = f_k + \alpha_k \odot f_k$$

  以防止特征过度抑制；

- **BatchNorm + ReLU**：稳定梯度与训练收敛；

- **可解释性增强**：保留并输出每层 gating map，用于后续可视化。

## 3. 改进细节与实现要点

| 模块                    | 改进点                           | 技术说明                               |
| ----------------------- | -------------------------------- | -------------------------------------- |
| **Backbone**            | 保留多尺度特征输出 f0–f4         | 使用 ResNet34 / 50 作为特征提取器      |
| **Transformer Encoder** | 输出 memory (B, C, Ht, Wt)       | 建模全局上下文语义                     |
| **TGSC**                | 上采样 memory → 门控各 skip 特征 | 完成全局语义对局部特征的动态调制       |
| **Decoder**             | 替换原 AttentionGate             | 使用 TGSC 替代，参数更少、语义传递更强 |
| **Edge Head**           | 输出 edge logits/prob            | 与原结构相同                           |

## 4. 模型评估与实验设计

### 4.1 主要任务

本模型针对 **语义引导的边缘检测任务（semantic-guided edge detection）**。 目标是：

- 抑制噪声纹理边缘；
- 提高物体真实边界的连续性；
- 实现语义一致的高分辨率边缘预测。

### 4.2 评价指标

- **ODS-F / OIS-F**：边缘检测常用指标；
- **Precision / Recall**：衡量预测与真值的匹配程度；
- **IoU / Boundary IoU**：用于语义一致性度量；
- **参数量 (M)、FLOPs (G)**：评估效率；
- **FPS**：推理速度；
- **Explainability metrics**（可选）：gating map 与 GT 边缘的 IoU。

## 5. 消融实验设计建议

| 实验类型       | 对比项                                                | 目的               |
| -------------- | ----------------------------------------------------- | ------------------ |
| **基础对比**   | (a) 无 gate 拼接；(b) AttentionGate；(c) TGSC (ours)  | 验证 TGSC 效果提升 |
| **变体对比**   | TGSC-additive / TGSC-spatial-only / TGSC-channel-only | 分析门控类型影响   |
| **投影共享性** | 共享 vs 非共享 memory 投影                            | 检验语义一致性影响 |
| **正则化**     | 加入稀疏正则 / 一致性损失                             | 验证 gate 稳定性   |
| **多任务扩展** | 边缘检测 → 分割任务                                   | 测试泛化与适用性   |

## 6. 同类模型比较建议

| 模型                         | 特征融合方式                  | 优点                 | TGSC 优势              |
| ---------------------------- | ----------------------------- | -------------------- | ---------------------- |
| **HED / DexiNed (CNN-only)** | 多层边缘监督，缺乏全局语义    | 精度高但误检多       | TGSC 抑制无关纹理      |
| **Attention U-Net**          | 局部门控 skip                 | 简单易实现           | TGSC 具全局引导能力    |
| **TransUNet / UCTransNet**   | Transformer + cross-attn 融合 | 强全局建模，计算量大 | TGSC 更轻量，参数少    |
| **Ours (TGSC)**              | 全局语义门控 skip             | 轻量化 + 语义一致    | 提升边缘连续性与抗噪性 |

## 7. 可解释性与可视化分析

为展示 TGSC 的有效性与可解释性，建议进行以下可视化实验：

1. **Gating Map 热力图**：展示 Transformer 如何调制不同 skip 层的特征；
2. **不同层的融合效果**：对比原特征、gated 特征及最终边缘输出；
3. **多尺度边缘一致性可视化**：说明 TGSC 如何减少断裂、增强目标轮廓；
4. **语义一致性分析**：定量测量 gating map 与真实边界区域的重叠率。

## 8. 模型改进的意义与创新点总结

| 维度         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| **核心创新** | 引入全局语义门控机制，使 Transformer 全局特征贯穿整个解码路径，实现全局-局部自适应融合 |
| **技术优势** | 实现简单、参数低、可解释性强、语义一致性高                   |
| **科研价值** | 弥合了 CNN-Transformer 混合结构中“语义传递中断”的长期瓶颈    |
| **应用前景** | 可广泛应用于边缘检测、语义分割、医学影像结构识别等需要“高分辨率结构 + 全局语义一致性”的任务 |

## 9. 下一步可选研究方向

- 将 TGSC 与多尺度 Transformer 交互模块结合，实现双路径增强；
- 引入跨模态语义门控（例如深度 + RGB）；
- 设计 TGSC 的可学习位置偏置，使 gating 与空间语义自适应；
- 引入轻量一致性损失（Consistency Loss）稳定 gating 学习。

**结语**： 本次改进的核心在于让 Transformer 的“语义理解”真正贯穿整个解码过程，实现从全局到局部的动态融合。TGSC 不仅能有效提升边缘检测精度与连续性，同时具备较强的可解释性和扩展潜力。