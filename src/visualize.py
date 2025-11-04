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
    Plots training and validation loss curves, and learning rate schedule.
    
    Args:
        log_data (Dict[str, List[float]]): Parsed training log data
        output_dir (str): Directory to save the plots
    """
    if not log_data:
        print("No training data to plot")
        return
    
    epochs = log_data['epochs']
    train_losses = log_data['train_losses']
    val_losses = log_data['val_losses']
    learning_rates = log_data['learning_rates']
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot loss curves
    ax1.plot(epochs[:len(train_losses)], train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot learning rate
    ax2.plot(epochs[:len(learning_rates)], learning_rates, 'g-', label='Learning Rate', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved training curves to {save_path}")

def plot_results(model, dataloader, device, num_samples: int, output_dir: str):
    """
    Selects random samples, runs inference, and plots a comparison of
    (Original Image, Ground Truth Edge, Predicted Edge).
    """
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a batch of data
    images, targets = next(iter(dataloader))
    images, targets = images.to(device), targets.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs['edge_prob']

    # Denormalize image for visualization
    mean = torch.tensor(project_config.data.mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(project_config.data.std).view(1, 3, 1, 1).to(device)
    images = images * std + mean

    # Plot and save
    for i in range(min(num_samples, images.size(0))):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.suptitle(f'Sample {i+1}', fontsize=16)

        # 1. Original Image
        img_display = TF.to_pil_image(images[i].cpu())
        axes[0].imshow(img_display)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 2. Ground Truth
        gt_display = targets[i, 0].cpu().numpy()
        axes[1].imshow(gt_display, cmap='gray')
        axes[1].set_title('Ground Truth Edge')
        axes[1].axis('off')

        # 3. Predicted Edge
        pred_display = preds[i, 0].cpu().numpy()
        axes[2].imshow(pred_display, cmap='gray')
        axes[2].set_title('Predicted Edge')
        axes[2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f'result_sample_{i+1}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved visualization to {save_path}")

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

    # Run visualization
    plot_results(model, dataloader, device, args.num_samples, args.output_dir)

if __name__ == '__main__':
    main()