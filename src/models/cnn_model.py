"""PyTorch Convolutional Neural Network (CNN) architectures for crop-stress diagnosis.

Includes:
1. Multispectral1DCNN: 1D Convolutional Neural Network operating on spectral-spatial
   biophysical feature vectors (NDVI, NDRE, GNDVI, SAVI, EVI, NDVI std, NDVI p10,
   low-vigor fraction, NDRE/NDVI ratio). Works across all sample kinds (rasters and CSV tables).
2. MultispectralSpatial2DCNN: 2D Convolutional Neural Network operating directly on
   the 5-band multispectral canopy rasters (Blue, Green, Red, Red-Edge, NIR) to extract
   spatial canopy textures, lesion clusters, gradients, and uniform deficit patterns.
3. Grad-CAM support for generating spatial class activation heatmaps across the field canopy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5
DEFAULT_FEATURE_DIM = 9
NUM_BANDS = 5


class Multispectral1DCNN(nn.Module):
    """1D Deep Convolutional Neural Network for spectral feature vectors.

    Architecture:
      Input (B, 1, 9)
      Conv1D(1 -> 32, k=3, p=1) -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.1)
      Conv1D(32 -> 64, k=3, p=1) -> BatchNorm1d -> LeakyReLU(0.1) -> MaxPool1d(2)
      Conv1D(64 -> 128, k=3, p=1) -> BatchNorm1d -> LeakyReLU(0.1) -> AdaptiveAvgPool1d(1)
      Linear(128 -> 64) -> LeakyReLU(0.1) -> Dropout(0.3)
      Linear(64 -> 5)
    """

    def __init__(self, in_features: int = DEFAULT_FEATURE_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout1d(0.1),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x: (batch_size, in_features) or (batch_size, 1, in_features)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        logits = self.classifier(x)
        return logits

    def compute_saliency(self, x: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int]:
        """Compute input-gradient feature attribution (saliency = d(logit)/dx * x)."""
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.clone().detach().requires_grad_(True)
        if x.dim() == 2:
            x_in = x.unsqueeze(1)
        else:
            x_in = x
        out = self.forward(x_in)
        if target_class is None:
            target_class = int(torch.argmax(out, dim=1).item())
        score = out[0, target_class]
        score.backward()
        grad = x.grad[0].cpu().numpy()
        x_np = x[0].detach().cpu().numpy()
        # Sensitivity attribution: grad * x
        saliency = grad * x_np
        return saliency, target_class


class MultispectralSpatial2DCNN(nn.Module):
    """2D Deep Convolutional Neural Network for multispectral 5-band canopy rasters.

    Architecture:
      Input (B, 5, H, W)
      Conv2d(5 -> 32, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
      Conv2d(32 -> 64, k=3, p=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
      Conv2d(64 -> 128, k=3, p=1) -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d((2, 2))
      Linear(128 * 2 * 2 -> 64) -> ReLU -> Dropout(0.3)
      Linear(64 -> 5)
    """

    def __init__(self, in_channels: int = NUM_BANDS, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        self._feature_maps = None
        self._gradients = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x: (batch_size, in_channels, H, W)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

    def generate_gradcam(self, x: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int]:
        """Generate Grad-CAM (Gradient-weighted Class Activation Map) on 2D raster."""
        self.eval()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h_orig, w_orig = x.shape[-2], x.shape[-1]

        # Hook to capture activations and gradients in block3
        feature_maps = []
        gradients = []

        target_layer = self.block3[0]  # Conv2d(64, 128)

        def fwd_hook(module, inp, output):
            feature_maps.append(output)

        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        try:
            x_var = x.clone().detach().requires_grad_(True)
            out = self.forward(x_var)
            if target_class is None:
                target_class = int(torch.argmax(out, dim=1).item())

            self.zero_grad()
            score = out[0, target_class]
            score.backward()

            f_map = feature_maps[0][0]  # (128, H', W')
            grads = gradients[0][0]     # (128, H', W')
            weights = torch.mean(grads, dim=(1, 2), keepdim=True)  # (128, 1, 1)

            cam = torch.sum(weights * f_map, dim=0)  # (H', W')
            cam = F.relu(cam)
            cam = cam.detach().cpu().numpy()

            # Normalize cam to [0, 1]
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros_like(cam)

            # Resize to original (H, W) using bilinear interpolation
            cam_tensor = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            cam_resized = F.interpolate(cam_tensor, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
            heatmap = cam_resized.squeeze().numpy()
            return heatmap, target_class
        finally:
            h1.remove()
            h2.remove()
