import os
from dataclasses import dataclass, field
from typing import List, Tuple

# --- 1. Path Configuration ---
@dataclass
class PathConfig:
    """Configuration for all data and output paths."""
    dataset_root: str = "c:/Users/Administrator/Desktop/CV_Paper_2/BSDS500-master/processed/"
    output_dir: str = "c:/Users/Administrator/Desktop/CV_Paper_2/output/"
    checkpoint_dir: str = field(init=False)

    def __post_init__(self):
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

# --- 2. Model Architecture Configuration ---
@dataclass
class ModelConfig:
    """Hyperparameters for the model architecture."""
    backbone: str = 'resnet34'  # resnet34 or resnet50
    pretrained: bool = True
    d_model: int = 256  # Transformer dimension
    nhead: int = 8  # Number of attention heads
    num_transformer_layers: int = 4
    dim_feedforward: int = 1024
    transformer_dropout: float = 0.1
    decoder_attention: bool = True  # Use attention gates in the decoder
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)
    edge_head_channels: int = 32

# --- 3. Data Loading and Augmentation Configuration ---
@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    dataset_root: str = "c:/Users/Administrator/Desktop/CV_Paper_2/BSDS500-master/processed/"
    resize_dim: Tuple[int, int] = (320, 320)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    num_workers: int = 8  # Increased for better data loading performance
    edge_threshold: float = 0.1 # Threshold to binarize edge maps for training
    # Augmentation
    random_flip: bool = True
    color_jitter: bool = True
    random_rotate: bool = True
    rotate_degrees: int = 10
    random_scale: bool = True
    scale_range: Tuple[float, float] = (0.9, 1.1)

# --- 4. Training Configuration ---
@dataclass
class TrainConfig:
    """Hyperparameters for the training process."""
    epochs: int = 500
    batch_size: int = 16  # Increased batch size for better GPU utilization
    gradient_accumulation_steps: int = 2  # Accumulate gradients over multiple steps
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    # Loss weights
    bce_weight: float = 1.0
    dice_weight: float = 0.5
    use_dice: bool = True
    # BCE weights (used when not using focal loss)
    bce_pos_weight: float = 8.0  # Stronger weight for edge pixels
    bce_neg_weight: float = 0.25  # Lower weight for non-edge pixels
    # Focal Loss settings
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # Optimization strategies
    warmup_epochs: int = 3
    grad_clip_norm: float = 1.0
    seed: int = 42
    # Early stopping
    early_stopping_patience: int = 10  # Number of epochs to wait before stopping
    early_stopping_min_delta: float = 0.001  # Minimum change to qualify as improvement

# --- 5. Evaluation Configuration ---
@dataclass
class EvalConfig:
    """Configuration for the evaluation process."""
    num_vis_samples: int = 10 # Number of images to save in visualization

# --- 6. Project-wide Config ---
@dataclass
class ProjectConfig:
    path: PathConfig
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig

# Instantiate and export the global config object
project_config = ProjectConfig(
    path=PathConfig(),
    model=ModelConfig(),
    data=DataConfig(),
    train=TrainConfig(),
    eval=EvalConfig()
)