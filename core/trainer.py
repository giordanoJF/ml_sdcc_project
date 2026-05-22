"""Training and validation utilities for the local worker loop."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple,
    device: torch.device,
    clip_grad: float = 1.0,
    label_smoothing: float = 0.1,
) -> float:
    """
    Run a single forward-backward-update step.
    Returns the scalar loss value for logging purposes.

    clip_grad:       max L2 norm for gradient clipping; limits client drift in FL
    label_smoothing: softens hard targets (0.1 spread over 62 classes)
    """
    x, y = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y, label_smoothing=label_smoothing)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
    optimizer.step()
    return loss.item()


def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on the local validation set.
    Returns (avg_loss, accuracy) over the full validation split.
    Uses reduction='sum' to accumulate correctly across batches.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    model.train()
    if total == 0:
        return float("inf"), 0.0
    return total_loss / total, correct / total
