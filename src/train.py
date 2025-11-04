"""
Main training script for the Edge-Aware Transformer model.
"""
import os
import torch
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import logging

# For mixed precision training


from config import project_config
from model import build_model
from dataset import create_dataloader
from loss import EdgeDetectionLoss

# --- Setup ---
def setup_logging(log_path: str):
    """Configures logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run the training and validation process."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Edge Detection Model Training')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume training from the best checkpoint')
    args = parser.parse_args()
    
    # Configs
    cfg = project_config
    output_dir = cfg.path.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cfg.path.checkpoint_dir, exist_ok=True)

    # Logging
    setup_logging(os.path.join(output_dir, 'train.log'))
    logging.info("--- Starting Edge Detection Model Training ---")
    logging.info(f"Configuration loaded. Output directory: {output_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data ---
    logging.info("Creating data loaders...")
    train_loader = create_dataloader(cfg.data, cfg.train, split='train', augment=True)
    val_loader = create_dataloader(cfg.data, cfg.train, split='val', augment=False)
    logging.info(f"Train loader: {len(train_loader)} batches | Val loader: {len(val_loader)} batches")

    # --- Model, Loss, Optimizer ---
    logging.info("Building model...")
    model = build_model(cfg.model).to(device)
    
    # Check if we should resume from checkpoint
    best_checkpoint_path = os.path.join(cfg.path.checkpoint_dir, 'best_model.pth')
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(best_checkpoint_path):
        logging.info(f"Resuming training from checkpoint: {best_checkpoint_path}")
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        logging.info("Loaded best model weights successfully")
    else:
        if args.resume:
            logging.warning("Checkpoint not found, starting training from scratch")
        else:
            logging.info("Starting training from scratch")
    
    criterion = EdgeDetectionLoss(cfg.train).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Mixed precision training scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- Training Loop ---
    logging.info("Starting training loop...")
    
    # Early stopping variables
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(start_epoch, cfg.train.epochs):
        logging.info(f"\n--- Epoch {epoch+1}/{cfg.train.epochs} ---")

        # 1. Training Phase
        model.train()
        train_loss_total = 0.0
        accumulation_steps = cfg.train.gradient_accumulation_steps
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        
        from contextlib import nullcontext
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(device), targets.to(device)

            # Mixed precision training
            with (torch.amp.autocast('cuda') if device.type == 'cuda' else nullcontext()):
                outputs = model(images)
                loss_dict = criterion(outputs['edge_logits'], targets)
                loss = loss_dict['total_loss']
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling for mixed precision
            if device.type == 'cuda':
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights only after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if device.type == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_loss_total += loss.item() * accumulation_steps
            progress_bar.set_postfix(loss=loss.item() * accumulation_steps)
        
        avg_train_loss = train_loss_total / len(train_loader)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Get GPU memory usage if using CUDA
        gpu_memory_info = ""
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
            gpu_memory_info = f" | GPU Mem: {allocated:.2f}/{reserved:.2f} GB"
        
        logging.info(f"Average Training Loss: {avg_train_loss:.4f} | LR: {current_lr:.2e}{gpu_memory_info}")

        # 2. Validation Phase
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False)
            for images, targets in progress_bar_val:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss_dict = criterion(outputs['edge_logits'], targets)
                loss = loss_dict['total_loss']
                val_loss_total += loss.item()
                progress_bar_val.set_postfix(loss=loss.item())

        avg_val_loss = val_loss_total / len(val_loader)
        logging.info(f"Average Validation Loss: {avg_val_loss:.4f}")

        # Log additional epoch information
        if device.type == 'cuda':
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
            max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3    # GB
            logging.info(f"Max GPU Memory Usage: {max_allocated:.2f}/{max_reserved:.2f} GB")
        
        # Reset memory stats for next epoch
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        # 3. Scheduler and Checkpointing
        scheduler.step(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_val_loss - cfg.train.early_stopping_min_delta:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(cfg.path.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Validation loss improved. Saved best model to {checkpoint_path}")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            logging.info(f"No improvement in validation loss for {patience_counter} epochs")
            
            # Check if we should stop early
            if patience_counter >= cfg.train.early_stopping_patience:
                logging.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        # Clear GPU cache to free memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    logging.info("--- Training Finished ---")

if __name__ == '__main__':
    main()