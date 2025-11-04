"""
PyTorch Dataset and DataLoader for the preprocessed BSDS500 dataset.
"""
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from typing import Tuple, List, Optional

from config import DataConfig, TrainConfig

class BSDS500Dataset(Dataset):
    """
    PyTorch Dataset for the preprocessed BSDS500 data.
    Loads images and their corresponding edge maps.
    """
    def __init__(self, config: DataConfig, split: str = 'train', augment: bool = False):
        """
        Args:
            config (DataConfig): Configuration object with data paths and parameters.
            split (str): The dataset split to use ('train', 'val', or 'test').
            augment (bool): Whether to apply data augmentation.
        """
        super().__init__()
        self.config = config
        self.split = split
        self.augment = augment
        self.root_dir = config.dataset_root
        
        # Load metadata
        metadata_path = os.path.join(self.root_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please run the data processing script first.")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get file lists for the specified split
        # Find the split configuration in the splits list
        split_config = None
        for split_item in metadata['splits']:
            if split_item['split'] == self.split:
                split_config = split_item
                break
        
        if split_config is None:
            raise ValueError(f"Split '{self.split}' not found in metadata. Available splits: {[s['split'] for s in metadata['splits']]}")
        
        # Load the list of files from the split's text file
        # Each line contains: image_path edge_path
        with open(split_config['list_file'], 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Parse image and edge file paths from each line
        self.image_files = []
        self.edge_files = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                self.image_files.append(parts[0])
                self.edge_files.append(parts[1])
        
        # Define transformations
        self.image_transform, self.edge_transform = self._get_transformations()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads and transforms a single image and its edge map."""
        # File paths
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        edge_path = os.path.join(self.root_dir, self.edge_files[idx])
        
        # Load data
        image = Image.open(img_path).convert('RGB')
        edge = Image.open(edge_path).convert('L') # Grayscale
        
        # Apply transformations
        if self.image_transform and self.edge_transform:
            state = torch.get_rng_state()
            image = self.image_transform(image)
            torch.set_rng_state(state)
            edge = self.edge_transform(edge)

        # Edge tensor from ToTensor() is already in [0,1]; binarize using configured threshold
        edge = (edge >= self.config.edge_threshold).float()

        return image, edge

    def _get_transformations(self) -> Tuple[T.Compose, T.Compose]:
        """Construct transformations for images and edges with proper interpolation.
        Images use bilinear interpolation and normalization; edges use nearest interpolation.
        """
        image_transforms: List = []
        edge_transforms: List = []

        # Resize with different interpolation for image vs. edge
        image_transforms.append(T.Resize(self.config.resize_dim, interpolation=T.InterpolationMode.BILINEAR))
        edge_transforms.append(T.Resize(self.config.resize_dim, interpolation=T.InterpolationMode.NEAREST))

        # Augmentations (training only)
        if self.split == 'train' and self.augment:
            if self.config.random_flip:
                image_transforms.append(T.RandomHorizontalFlip())
                edge_transforms.append(T.RandomHorizontalFlip())
            # Random affine (rotation/scale) with synchronized RNG in __getitem__
            if getattr(self.config, 'random_rotate', False) or getattr(self.config, 'random_scale', False):
                degrees = self.config.rotate_degrees if getattr(self.config, 'random_rotate', False) else 0
                scale = self.config.scale_range if getattr(self.config, 'random_scale', False) else None
                image_transforms.append(T.RandomAffine(degrees=degrees, scale=scale, interpolation=T.InterpolationMode.BILINEAR))
                edge_transforms.append(T.RandomAffine(degrees=degrees, scale=scale, interpolation=T.InterpolationMode.NEAREST))
            if self.config.color_jitter:
                image_transforms.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))

        # Finalize pipelines
        image_transforms.append(T.ToTensor())
        image_transforms.append(T.Normalize(mean=self.config.mean, std=self.config.std))
        edge_transforms.append(T.ToTensor())

        return T.Compose(image_transforms), T.Compose(edge_transforms)

def create_dataloader(data_config: DataConfig, train_config: TrainConfig, split: str, augment: bool) -> DataLoader:
    """
    Creates a DataLoader for a specific dataset split.

    Args:
        data_config (DataConfig): Configuration for data loading.
        train_config (TrainConfig): Configuration for training (e.g., batch size).
        split (str): The dataset split ('train', 'val', or 'test').
        augment (bool): Whether to apply data augmentation.

    Returns:
        DataLoader: The configured PyTorch DataLoader.
    """
    dataset = BSDS500Dataset(config=data_config, split=split, augment=augment)
    
    shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=shuffle,
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=shuffle # Drop last incomplete batch only for training
    )
    return dataloader