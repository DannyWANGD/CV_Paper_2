"""
Edge-Aware Transformer Model Components
- CNN Backbone (ResNet)
- Transformer Encoder
- U-Net Style Decoder
- Edge Head
- Full Model Assembly
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet50_Weights
from typing import Dict, List, Tuple

from config import ModelConfig

# --- 1. CNN Backbone ---
class ResNetBackbone(nn.Module):
    """ResNet backbone to extract multi-scale features."""
    def __init__(self, backbone_name: str = 'resnet34', pretrained: bool = True):
        super().__init__()
        if backbone_name == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet34(weights=weights)
            self.out_channels = [64, 64, 128, 256, 512]
        elif backbone_name == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet50(weights=weights)
            self.out_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features at different stages."""
        f0 = self.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(self.maxpool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {'f0': f0, 'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4}

# --- 2. Transformer Encoder ---
class TransformerEncoder(nn.Module):
    """Transformer encoder to model global relationships."""
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """Forward pass with positional encoding."""
        return self.transformer_encoder(src + pos_embed)

# --- 3. Decoder Components ---
class AttentionGate(nn.Module):
    """Attention Gate to focus on relevant skip-connection features."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DecoderBlock(nn.Module):
    """A single block in the U-Net style decoder."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if use_attention:
            self.attn = AttentionGate(in_channels, skip_channels, (in_channels + skip_channels) // 2)
        
        conv_in_channels = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.use_attention:
            skip = self.attn(g=x, x=skip)
        
        # Ensure spatial dimensions match
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# --- 4. Edge Head ---
class EdgeHead(nn.Module):
    """Predicts edge map from decoder features."""
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# --- 5. Full Model ---
class EdgeAwareTransformer(nn.Module):
    """The complete Edge-Aware Transformer model."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 1. Backbone
        self.backbone = ResNetBackbone(config.backbone, config.pretrained)
        backbone_channels = self.backbone.out_channels
        
        # 2. Input projection for Transformer
        self.input_proj = nn.Conv2d(backbone_channels[-1], config.d_model, kernel_size=1)
        
        # 3. Positional Encoding
        # We create this dynamically in the forward pass based on input size
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, config.d_model)) # Placeholder size
        
        # 4. Transformer Encoder
        self.transformer = TransformerEncoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_transformer_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout
        )
        
        # 5. Decoder
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(config.d_model, backbone_channels[3], config.decoder_channels[0], config.decoder_attention),
            DecoderBlock(config.decoder_channels[0], backbone_channels[2], config.decoder_channels[1], config.decoder_attention),
            DecoderBlock(config.decoder_channels[1], backbone_channels[1], config.decoder_channels[2], config.decoder_attention),
            DecoderBlock(config.decoder_channels[2], backbone_channels[0], config.decoder_channels[3], config.decoder_attention)
        ])
        
        # 6. Edge Head
        self.edge_head = EdgeHead(config.decoder_channels[3], config.edge_head_channels, 1)

    def _generate_positional_encoding(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Generate 2D sinusoidal positional encoding."""
        # Create grid positions
        pos_x = torch.arange(w, device=device, dtype=torch.float32)
        pos_y = torch.arange(h, device=device, dtype=torch.float32)
        
        # Create 2D grid
        grid_x, grid_y = torch.meshgrid(pos_x, pos_y, indexing='xy')
        
        # Flatten grid positions
        pos = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (h*w, 2)
        
        # Generate sinusoidal encoding
        dim_t = torch.arange(self.config.d_model // 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (2 * dim_t / self.config.d_model))
        
        # Position encoding for x and y coordinates
        pos_enc = torch.zeros(h * w, self.config.d_model, device=device)
        
        # Even indices: sine, odd indices: cosine
        pos_enc[:, 0::2] = torch.sin(pos[:, 0:1] * inv_freq)  # x coordinate
        pos_enc[:, 1::2] = torch.cos(pos[:, 0:1] * inv_freq)  # x coordinate
        pos_enc[:, 0::2] += torch.sin(pos[:, 1:2] * inv_freq)  # y coordinate
        pos_enc[:, 1::2] += torch.cos(pos[:, 1:2] * inv_freq)  # y coordinate
        
        return pos_enc.unsqueeze(0)  # (1, h*w, d_model)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[2:]
        
        # Backbone features
        features = self.backbone(x)
        f0, f1, f2, f3, f4 = features['f0'], features['f1'], features['f2'], features['f3'], features['f4']
        
        # Project and flatten for Transformer
        proj = self.input_proj(f4)
        bs, c, h, w = proj.shape
        proj_flat = proj.flatten(2).permute(0, 2, 1)  # B x (H*W) x C
        
        # Positional encoding
        pos_embed = self._generate_positional_encoding(h, w, x.device)
        
        # Transformer encoding
        memory = self.transformer(proj_flat, pos_embed)
        memory = memory.permute(0, 2, 1).view(bs, c, h, w)
        
        # Decoder path
        d = memory
        d = self.decoder_blocks[0](d, f3)
        d = self.decoder_blocks[1](d, f2)
        d = self.decoder_blocks[2](d, f1)
        d = self.decoder_blocks[3](d, f0)
        
        # Edge prediction
        edge_logits = self.edge_head(d)
        edge_logits = F.interpolate(edge_logits, size=input_shape, mode='bilinear', align_corners=True)
        
        return {
            'edge_logits': edge_logits,
            'edge_prob': torch.sigmoid(edge_logits)
        }

# --- Helper to build model ---
def build_model(config: ModelConfig) -> EdgeAwareTransformer:
    """Builds the model from configuration."""
    model = EdgeAwareTransformer(config)
    return model