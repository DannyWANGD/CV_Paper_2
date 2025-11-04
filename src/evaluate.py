"""
Evaluation script for the Edge-Aware Transformer model.
Calculates BSDS metrics (ODS, OIS, AP) and saves predictions.
"""
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from sklearn.metrics import precision_recall_curve, average_precision_score
from typing import Tuple

from config import project_config
from model import build_model
from dataset import create_dataloader

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

def pr_curve_to_f_score(precision: np.ndarray, recall: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Calculates F-score from precision and recall."""
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)

def main():
    """Main function to run the evaluation."""
    cfg = project_config
    output_dir = cfg.path.output_dir
    eval_dir = os.path.join(output_dir, 'evaluation')
    pred_dir = os.path.join(eval_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    setup_logging(os.path.join(eval_dir, 'eval.log'))
    logging.info("--- Starting Edge Detection Model Evaluation ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data ---
    logging.info("Creating test data loader...")
    test_loader = create_dataloader(cfg.data, cfg.train, split='test', augment=False)

    # --- Model ---
    logging.info("Building model...")
    model = build_model(cfg.model).to(device)
    checkpoint_path = os.path.join(cfg.path.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return
    # Load weights securely; avoids unsafe pickle execution
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    logging.info(f"Model loaded from {checkpoint_path}")

    # --- Evaluation ---
    all_preds = []
    all_targets = []
    ois_f_scores = []

    logging.info("Running inference on the test set...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            outputs = model(images)
            preds = outputs['edge_prob'].cpu().numpy()
            targets = targets.cpu().numpy()

            for j in range(preds.shape[0]):
                pred_single = preds[j, 0]
                target_single = targets[j, 0]

                # Save prediction
                pred_img = Image.fromarray((pred_single * 255).astype(np.uint8))
                pred_img.save(os.path.join(pred_dir, f'test_{i*cfg.train.batch_size + j}.png'))

                # For ODS
                all_preds.append(pred_single.flatten())
                all_targets.append(target_single.flatten())

                # For OIS
                precision, recall, _ = precision_recall_curve(target_single.flatten(), pred_single.flatten())
                f_scores = pr_curve_to_f_score(precision, recall)
                ois_f_scores.append(np.max(f_scores) if len(f_scores) > 0 else 0)

    # --- Calculate Metrics ---
    logging.info("Calculating metrics...")
    
    # ODS (Optimal Dataset Scale)
    all_preds_flat = np.concatenate(all_preds)
    all_targets_flat = np.concatenate(all_targets)
    pos_count = int(np.sum(all_targets_flat))
    total_count = int(all_targets_flat.shape[0])
    neg_count = total_count - pos_count
    pred_min, pred_max, pred_mean = float(np.min(all_preds_flat)), float(np.max(all_preds_flat)), float(np.mean(all_preds_flat))

    logging.info(f"Label stats — pos: {pos_count}, neg: {neg_count}, pos_ratio: {pos_count/total_count:.4f}")
    logging.info(f"Pred stats — min: {pred_min:.6f}, max: {pred_max:.6f}, mean: {pred_mean:.6f}")

    if pos_count == 0:
        logging.warning("Test labels contain no positive edges after thresholding. Check DataConfig.edge_threshold and dataset.")

    precision_ods, recall_ods, thresholds_ods = precision_recall_curve(all_targets_flat, all_preds_flat)
    # Align thresholds with precision/recall (sklearn returns thresholds of length N-1)
    f_scores_all = pr_curve_to_f_score(precision_ods, recall_ods)
    f_scores_thr = f_scores_all[1:] if len(f_scores_all) > 1 else f_scores_all
    if len(thresholds_ods) > 0 and len(f_scores_thr) > 0:
        i_best = int(np.argmax(f_scores_thr))
        ods_f = float(f_scores_thr[i_best])
        best_threshold = float(thresholds_ods[i_best])
    else:
        ods_f = 0.0
        best_threshold = 0.0

    # OIS (Optimal Image Scale)
    ois_f = np.mean(ois_f_scores)

    # AP (Average Precision)
    # Average Precision (area under precision-recall curve)
    try:
        ap = float(average_precision_score(all_targets_flat, all_preds_flat))
    except Exception:
        ap = 0.0

    logging.info("--- Evaluation Results ---")
    logging.info(f"ODS F-Score: {ods_f:.4f} (at threshold {best_threshold:.4f})")
    logging.info(f"OIS F-Score: {ois_f:.4f}")
    logging.info(f"Average Precision (AP): {ap:.4f}")
    logging.info(f"Predictions saved to: {pred_dir}")

if __name__ == '__main__':
    main()