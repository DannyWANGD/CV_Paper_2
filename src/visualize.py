"""
Visualization utilities for the Edge-Aware Transformer model.
"""
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Dict, List, Tuple

from config import project_config
from model import build_model
from dataset import create_dataloader

def parse_training_log(log_path: str) -> Dict[str, List[float]]:
    """
    Parses the training log file to extract loss values and other metrics.
    
    Args:
        log_path (str): Path to the training log file
        
    Returns:
        Dict[str, List[float]]: Dictionary containing training and validation losses
    """
    train_losses = []
    val_losses = []
    learning_rates = []
    epochs = []
    
    if not os.path.exists(log_path):
        print(f"Training log file not found at {log_path}")
        return {}
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # Parse training loss
        if "Average Training Loss:" in line:
            match = re.search(r'Average Training Loss: (\d+\.\d+)', line)
            if match:
                train_losses.append(float(match.group(1)))
        
        # Parse validation loss
        if "Average Validation Loss:" in line:
            match = re.search(r'Average Validation Loss: (\d+\.\d+)', line)
            if match:
                val_losses.append(float(match.group(1)))
        
        # Parse learning rate
        if "LR:" in line:
            match = re.search(r'LR: ([\d\.e-]+)', line)
            if match:
                learning_rates.append(float(match.group(1)))
        
        # Parse epoch number
        if "Epoch" in line and "/" in line:
            match = re.search(r'Epoch (\d+)/', line)
            if match:
                epochs.append(int(match.group(1)))
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates
    }

def plot_training_curves(log_data: Dict[str, List[float]], output_dir: str):
    """
    绘制训练与验证损失曲线、学习率曲线，并在图与控制台中明确数据选区是否匹配。

    Args:
        log_data (Dict[str, List[float]]): 解析后的训练日志数据
        output_dir (str): 保存图像的目录
    """
    if not log_data:
        print("No training data to plot")
        return

    epochs = log_data.get('epochs', [])
    train_losses = log_data.get('train_losses', [])
    val_losses = log_data.get('val_losses', [])
    learning_rates = log_data.get('learning_rates', [])

    total_epochs = len(epochs)
    n_train = min(total_epochs, len(train_losses))
    n_val = min(total_epochs, len(val_losses))
    n_lr = min(total_epochs, len(learning_rates))

    # 控制台输出数据一致性检查
    print(
        (
            f"[Curves] Parsed epochs: {total_epochs}, "
            f"train points: {n_train}/{len(train_losses)}, "
            f"val points: {n_val}/{len(val_losses)}, "
            f"lr points: {n_lr}/{len(learning_rates)}"
        )
    )
    if (n_train < len(train_losses)) or (n_val < len(val_losses)) or (n_lr < len(learning_rates)):
        print("[Curves] Warning: lengths mismatch. Plotting with truncated aligned ranges.")

    # 对齐数据长度以避免错位
    epochs_train = epochs[:n_train]
    epochs_val = epochs[:n_val]
    epochs_lr = epochs[:n_lr]

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 绘制损失曲线
    if n_train > 0:
        ax1.plot(epochs_train, train_losses[:n_train], 'b-', label='Training Loss', linewidth=2)
    if n_val > 0:
        ax1.plot(epochs_val, val_losses[:n_val], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制学习率曲线（对齐长度）
    if n_lr > 0:
        ax2.plot(epochs_lr, learning_rates[:n_lr], 'g-', label='Learning Rate', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 在图中加入数据校验摘要
    summary_text = (
        f"Epochs: {total_epochs}\n"
        f"Train points: {n_train}/{len(train_losses)}\n"
        f"Val points: {n_val}/{len(val_losses)}\n"
        f"LR points: {n_lr}/{len(learning_rates)}"
    )
    fig.text(0.02, 0.02, summary_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    plt.tight_layout()

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved training curves to {save_path}")

def plot_composite(model, dataloader, device, output_dir: str, num_samples: int = 1, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.8):
    """
    Generate a single composite visualization containing:
    - Original image
    - Ground Truth edge map
    - Original image overlaid with colored predicted contour intensity (heatmap)
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # 预计算反归一化参数
    mean = torch.tensor(project_config.data.mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(project_config.data.std).view(1, 3, 1, 1).to(device)

    saved = 0
    for batch_idx, (images, targets) in enumerate(dataloader):
        if saved >= num_samples:
            break

        images, targets = images.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(images)
            preds = outputs['edge_prob']

        # 反归一化以便展示
        images = images * std + mean

        bsz = images.shape[0]
        for i in range(bsz):
            if saved >= num_samples:
                break

            img_pil = TF.to_pil_image(images[i].cpu())
            img_np = np.array(img_pil)  # HWC, uint8

            gt_np = targets[i, 0].cpu().numpy()
            pred_np = preds[i, 0].cpu().numpy()

            # 二值化阈值：优先使用 DataConfig.edge_threshold，否则默认 0.5
            th = getattr(project_config.data, 'edge_threshold', None)
            if th is None:
                th = 0.5
            bin_pred = (pred_np >= th).astype(np.uint8)

            # 构造 RGBA 叠加层（显眼颜色，如红色），仅在边缘像素处可见（保留为可选叠加参考）
            h, w = bin_pred.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            overlay[..., 0] = color[0]
            overlay[..., 1] = color[1]
            overlay[..., 2] = color[2]
            overlay[..., 3] = (bin_pred * int(alpha * 255))

            # 绘制综合图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            plt.suptitle('Composite Visualization (Image / GT / Image + Pred Heatmap)', fontsize=16)

            # 1. 原始图片
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # 2. Ground Truth 边缘图
            axes[1].imshow(gt_np, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            # 3. 原图 + 预测概率热力图叠加
            axes[2].imshow(img_np)
            pred_norm = np.clip(pred_np, 0.0, 1.0)
            heat = axes[2].imshow(pred_norm, cmap='turbo', alpha=0.6, vmin=0.0, vmax=1.0)
            axes[2].set_title('Image + Predicted Contour (colored intensity)')
            cbar = plt.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label('Edge Probability', rotation=90)
            axes[2].axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fname = f'composite_{saved+1:03d}.png'
            save_path = os.path.join(output_dir, fname)
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            saved += 1
            print(f"Saved composite visualization to {save_path}")

def main():
    """Main function to run visualization."""
    import argparse
    parser = argparse.ArgumentParser(description='Visualize model predictions and training curves')
    # Use project_config paths by default; allow overriding via CLI
    default_checkpoint = os.path.join(project_config.path.checkpoint_dir, 'best_model.pth')
    default_output_dir = os.path.join(project_config.path.output_dir, 'visualizations')
    default_log_file = os.path.join(project_config.path.output_dir, 'train.log')

    parser.add_argument('--checkpoint', type=str, default=default_checkpoint,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=15,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='Directory to save visualizations')
    parser.add_argument('--log_file', type=str, default=default_log_file,
                        help='Path to training log file')
    parser.add_argument('--plot_curves', action='store_true',
                        help='Plot training curves instead of sample predictions')
    # Add composite flag to avoid CLI errors and allow explicit selection
    parser.add_argument('--composite', action='store_true',
                        help='Generate composite visualization images (default behavior)')
    args = parser.parse_args()

    # Use project_config directly (no external YAML merging in this codebase)
    cfg = project_config

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.plot_curves:
        # Plot training curves
        log_data = parse_training_log(args.log_file)
        if log_data:
            plot_training_curves(log_data, args.output_dir)
        else:
            print("No training data found to plot curves")
        return

    # Create model
    model = build_model(cfg.model)
    model.to(device)

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        # Training saves pure state_dict, so load directly
        if isinstance(checkpoint, dict) and all(k.startswith(('conv', 'bn', 'layer', 'backbone', 'input_proj', 'decoder_blocks', 'edge_head')) or k.endswith(('weight', 'bias')) for k in checkpoint.keys()):
            model.load_state_dict(checkpoint)
        else:
            # Fallback in case a different format was saved
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Checkpoint not found at {args.checkpoint}")
        return

    # Create dataloader
    dataloader = create_dataloader(cfg.data, cfg.train, split='test', augment=False)

    # Run visualization — 支持批量生成指定数量的综合图
    plot_composite(model, dataloader, device, args.output_dir, num_samples=args.num_samples)

if __name__ == '__main__':
    main()