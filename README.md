# Edge-Aware Transformer（CV_Paper_2）

一个基于 CNN + Transformer 的边缘感知模型，专注于高质量的边缘检测，并为后续分割任务提供更锐利的边界监督。工程包含完整的数据预处理、训练、评估与可视化流程，开箱即用。

---

## 目录

- 项目概览与特性
- 快速开始（安装 / 数据 / 训练 / 评估 / 可视化）
- 模型结构（Backbone、Transformer、Decoder、Edge Head）
- 配置说明（参数与位置）
- 数据处理与格式（BSDS500 预处理、splits、metadata）
- 训练细节（损失、优化器、混合精度、早停、日志）
- 评估指标（ODS / OIS / AP）与输出
- 可视化（样例预测与训练曲线）
- 优化方向与扩展建议
- 常见问题与排错
- 许可证与致谢

---

## 项目概览与特性

- 边缘检测任务：当前实现以边缘检测为主，输出 `B x 1 x H x W` 的边缘概率图（edge_prob），同时保留 logits（edge_logits）。
- 模型组合：ResNet Backbone + Transformer Encoder + U-Net 风格 Decoder + Attention Gate + Edge Head。
- 数据预处理：提供 BSDS500 的预处理脚本，将原始数据转为标准化的 `processed/` 结构（图像、边缘、splits、metadata）。
- 训练与评估：支持混合精度、梯度累积、学习率 warmup、ReduceLROnPlateau 调度、早停与最佳模型保存；计算 BSDS 指标（ODS/OIS/AP）。
- 可视化：支持训练曲线绘制与测试样例可视化（原图 / GT 边缘 / 预测边缘）。

---

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

建议使用 PyTorch 2.0+（工程中部分 `weights_only=True` 加载方式需要较新版本）。

### 2) 数据准备（BSDS500 预处理）

- 默认指向本仓库中的示例路径（Windows）：
  - 原始数据：`c:/Users/Administrator/Desktop/CV_Paper_2/BSDS500-master/BSDS500/data`
  - 预处理输出：`c:/Users/Administrator/Desktop/CV_Paper_2/BSDS500-master/processed`

可以直接运行：

```bash
python data_process.py --thin --resize 320,320
```

可选参数：
- `--combine-mode {mean|max|or}`：多标注者的边界合并策略（默认 mean）
- `--edge-threshold 0.5`：可选边界二值化阈值（0~1），不设置则保留概率图
- `--thin`：对二值边缘图进行骨架化（细化）
- `--resize W,H`：图像与标签统一缩放尺寸（例如 320,320）

完成后会生成：
- `processed/images/{train|val|test}/...`
- `processed/edges/{train|val|test}/...`
- `processed/splits/{train|val|test}.txt`
- `processed/metadata.json`（包含路径与样本统计信息）

> 注意：训练代码仅使用 `processed/` 中的数据；不要删除原始 `BSDS500/data` 与 `bench/`（保留以便将来重新处理或对比评估）。

### 3) 训练

```bash
python src/train.py --epochs 100 --batch-size 16 --lr 1e-3 --dice-weight 0.5
```

常用可选项：
- `--resume`：如存在 `best_model.pth`，从最佳模型继续训练
- `--epochs`：覆盖 `TrainConfig.epochs`
- `--batch-size`：覆盖 `TrainConfig.batch_size`
- `--lr`：覆盖 `TrainConfig.learning_rate`
- `--dice-weight`：覆盖 `TrainConfig.dice_weight`

输出：
- 训练日志：`<output_dir>/train.log`
- 最佳模型：`<checkpoint_dir>/best_model.pth`
- 最新模型：`<checkpoint_dir>/last_model.pth`

### 4) 评估

```bash
python src/evaluate.py
```

输出：
- 预测图：`<output_dir>/evaluation/predictions/*.png`
- 评估日志：`<output_dir>/evaluation/eval.log`
- 指标：ODS / OIS / AP（在日志中输出与统计）

### 5) 可视化

- 样例预测（原图 / GT / 预测）：

```bash
python src/visualize.py --num_samples 15
```

- 训练曲线（loss 与学习率）：

```bash
python src/visualize.py --plot_curves
```

输出保存在：`<output_dir>/visualizations/`（如 `training_curves.png` 等）。

---

## 模型结构

文件：`src/model.py`

- Backbone（ResNet34/50）：来自 `torchvision`，可加载 ImageNet 预训练权重（`ModelConfig.pretrained=True`）。输出多尺度特征 `f0..f4`。
- Transformer Encoder：将最高层特征投影为 token（`1x1 conv` 到 `d_model`），加 2D 正弦位置编码，使用 `nn.TransformerEncoder` 建模全局关系。
- U-Net 风格 Decoder：逐步上采样并与 `f3/f2/f1/f0` 做跳跃连接；支持 Attention Gate（可开关）。
- Edge Head：两层 `Conv3x3+BN+ReLU` 后接 `Conv1x1` 输出边缘 logits（1 通道）。
- 输出：
  - `edge_logits`（未归一化）
  - `edge_prob`（`sigmoid(edge_logits)`）

> 提示：文档中提到的分割 Head 作为扩展方向；当前代码仅实现边缘检测分支。后续可按设计文档添加分割 Head 并在训练中加入联合损失。

---

## 配置说明（参数与位置）

文件：`src/config.py`

- `PathConfig`：工程路径（输出目录、checkpoint 目录等）
- `ModelConfig`：模型结构参数
  - `backbone`: `resnet34` 或 `resnet50`
  - `pretrained`: 是否加载 ImageNet 预训练
  - `d_model`/`nhead`/`num_transformer_layers`/`dim_feedforward`/`transformer_dropout`
  - `decoder_attention`: Decoder 跳跃连接是否启用 Attention Gate
  - `decoder_channels`: 解码器通道配置，如 `(256,128,64,32)`
  - `edge_head_channels`: 边缘分支中间通道数
- `DataConfig`：数据与增强参数
  - `dataset_root`: 指向 `processed/` 根路径（默认为本仓库示例路径）
  - `resize_dim`: 输入尺寸（默认 320×320）
  - `mean`/`std`: 归一化统计（ImageNet）
  - `num_workers`: DataLoader 线程数（Windows 可适当调低）
  - `edge_threshold`: 训练时二值化阈值（如需）
  - `random_flip`/`random_rotate`/`rotate_degrees`/`random_scale`/`scale_range`
- `TrainConfig`：训练参数与策略
  - `epochs`/`batch_size`/`learning_rate`/`weight_decay`
  - 损失：`bce_weight`、`dice_weight`、`use_dice`、`use_focal_loss`、`focal_alpha`、`focal_gamma`
  - BCE 权重：`bce_pos_weight`（正样本边缘权重）、`bce_neg_weight`
  - 优化：`warmup_epochs`、`grad_clip_norm`、`gradient_accumulation_steps`
  - 复现：`seed`
  - 早停：`early_stopping_patience`、`early_stopping_min_delta`
- `EvalConfig`：可视化样本数等
- `project_config`：全局配置对象（训练/评估/可视化脚本统一使用）

---

## 数据处理与格式（BSDS500）

文件：`data_process.py`

- 输入：`BSDS500/data`（包含 `images/` 与 `groundTruth/` 的 `.mat`）
- 输出：到 `processed/`（含 `images/`、`edges/`、`splits/` 与 `metadata.json`）
- `metadata.json` 示例字段：
  - `dataset_root`、`output_dir`
  - `splits`: 每个 split 的 `count`、`list_file`、`images_dir`、`edges_dir`
- Dataset 加载逻辑：`src/dataset.py` 根据 `metadata.json` 中的 `splits/*.txt` 逐行读取图像与边缘路径，并按 `DataConfig` 做同步变换与归一化。

---

## 训练细节

文件：`src/train.py`、`src/loss.py`

- 损失函数：
  - `weighted_bce_loss`：对边缘（正样本）与非边缘（负样本）使用不同权重
  - `focal_loss_with_logits`：可选（默认启用 `use_focal_loss=True`）以缓解类别不均衡与难例问题
  - `dice_loss`：可选组合提升重叠度（`dice_weight`）
  - 最终：`total_loss = bce_weight * BCE/Focal + dice_weight * Dice`
- 优化策略：
  - `AdamW` 优化器
  - `ReduceLROnPlateau` 调度器（以验证集平均损失为触发）
  - Warmup：前若干 epoch 线性增大学习率
  - 梯度裁剪：`grad_clip_norm`
  - 梯度累积：小显存下有效提升有效 batch size
  - 混合精度：`torch.amp.autocast` + `GradScaler`（CUDA）
- 早停与检查点：
  - 验证损失提升则保存 `best_model.pth`
  - 每轮保存 `last_model.pth`
  - `early_stopping_patience` 与 `early_stopping_min_delta` 控制早停
- 日志与监控：
  - 训练日志关键行：`Average Training Loss`、`Average Validation Loss`、`LR`
  - GPU 监控：最大显存（allocated/reserved）

---

## 评估指标与输出

文件：`src/evaluate.py`

- 指标：
  - ODS（Optimal Dataset Scale）：全数据集合最佳层面 F-score
  - OIS（Optimal Image Scale）：按图像选择最佳阈值后求平均 F-score
  - AP（Average Precision）：基于 PR 曲线的面积（`average_precision_score`）
- 输出：
  - `evaluation/predictions/*.png`：保存每张测试图的预测边缘图
  - `eval.log`：记录统计与数据分布（正负样本数、概率范围等）

---

## 可视化

文件：`src/visualize.py`

- 样例预测：`plot_results`
  - 从 `test` split 中采样，展示 原图 / GT / 预测，并保存到 `visualizations/`
- 训练曲线：`plot_training_curves`
  - 从 `train.log` 解析并绘制 loss 与学习率曲线（log-scale）

---

## 优化方向与扩展建议（精选）

可参考文档：`edge_aware_transformer_segmentation_详细设计方案.md`

- Backbone 替换：Swin、ConvNeXt、EfficientNet（提升语义表示）
- Transformer 改进：局部窗口注意力（Swin-like）、线性注意（Performer/LinearAttention）、多尺度 token（PVT/SegFormer）
- Decoder 强化：ASPP / Pyramid Pooling、Dual Decoder（边界与语义双分支）
- 边缘监督策略：多尺度深监督、Boundary Attention 融合
- 损失与重采样：OHEM 难例挖掘、Boundary-aware Loss（如 Lovász/Boundary IoU）
- 后处理：CRF/Refinement Network 以矫正边界
- 轻量化与加速：量化/剪枝/蒸馏、高效注意力近似
- 数据层面：更干净/更薄的边缘 GT、合成数据扩充边界多样性

> 当前代码专注边缘检测。若要引入分割任务，可添加 Segmentation Head（`Conv3x3 -> BN -> ReLU -> Conv1x1 -> logits`），与 Edge Head 联合训练：`L_total = L_seg + λ * L_edge`。

---

## 常见问题与排错

- 依赖问题：
  - `data_process.py` 读取 `.mat` 依赖 `scipy`；骨架化需 `scikit-image`
- 路径问题（Windows）：
  - `DataConfig.dataset_root` 默认是绝对路径；若移动工程，请在 `src/config.py` 中更新路径或通过 CLI 指定预处理输入/输出
- DataLoader 卡顿：
  - 在 Windows 环境可将 `num_workers` 调低（如 4/2/0），并关闭过多的增强操作
- 预训练权重下载：
  - `ModelConfig.pretrained=True` 时会下载 `torchvision` 权重；若网络受限，可设为 `False`
- 显存不足：
  - 调小 `batch_size`，启用混合精度，增大 `gradient_accumulation_steps`
- 训练曲线空白：
  - 检查 `train.log` 中是否存在关键字段（`Average Training Loss`、`Average Validation Loss`、`LR`）；否则无法解析

---

## 目录结构（简化）

```
CV_Paper_2/
├─ BSDS500-master/
│  └─ processed/
│     └─ metadata.json
├─ requirements.txt
├─ data_process.py
├─ src/
│  ├─ config.py       # 所有配置（路径/模型/数据/训练/评估）
│  ├─ dataset.py      # BSDS500 预处理数据的 Dataset 与 DataLoader
│  ├─ model.py        # 模型各模块与组装
│  ├─ loss.py         # BCE/Focal + Dice 组合损失
│  ├─ train.py        # 训练与验证主循环
│  ├─ evaluate.py     # 评估与指标计算
│  └─ visualize.py    # 可视化
└─ README.md
```

---

## 许可证与致谢

- 本工程使用 `torch`/`torchvision`、`numpy`、`scipy`、`scikit-image`、`scikit-learn`、`Pillow`、`matplotlib`、`tqdm` 等开源依赖。
- 数据集来源：BSDS500（Berkeley Segmentation Dataset）；请遵循其原始许可证与使用规范。

---

## 联系与后续

如果你需要：
- 添加分割分支与联合训练模板
- 集成更多指标（如 F-measure、PR 曲线保存、边界 IoU）
- 做 Ablation 自动化实验脚本

欢迎提出需求，我将根据你的实验目标继续完善工程。