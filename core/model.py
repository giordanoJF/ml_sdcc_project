"""
CNN for FEMNIST classification.

Input:  (N, 1, 28, 28) — grayscale 28×28 images
Output: (N, 62)        — logits over 62 character classes

Architecture: two double-conv blocks (VGG-style) + fully-connected classifier.
  - same-padding convolutions preserve spatial dimensions until each MaxPool
  - BatchNorm after every conv stabilises activations and speeds convergence
  - Dropout2d (spatial) and Dropout prevent overfitting on the small
    non-i.i.d. local partitions each worker holds in the federated setting
"""
import torch.nn as nn


class FEMNISTModel(nn.Module):
    """
    VGG-style CNN for FEMNIST (28×28 grayscale, 62 classes).

    Spatial flow:
        (N,  1, 28, 28)   input
        (N, 32, 28, 28)   after block1 conv layers  (same padding, no shrink)
        (N, 32, 14, 14)   after block1 MaxPool
        (N, 64, 14, 14)   after block2 conv layers  (same padding, no shrink)
        (N, 64,  7,  7)   after block2 MaxPool
        (N,     3136)     flatten  (64 × 7 × 7)
        (N,      512)     after fc1
        (N,       62)     logits (no softmax — use cross_entropy loss)
    """

    def __init__(
        self,
        num_classes: int = 62,
        dropout_conv: float = 0.25,
        dropout_fc: float = 0.5,
    ):
        super().__init__()

        # Block 1: 1 → 32 channels, spatial 28×28 → 14×14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_conv),
        )

        # Block 2: 32 → 64 channels, spatial 14×14 → 7×7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_conv),
        )

        # Classifier: 64×7×7 = 3136 → 512 → 62
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)
