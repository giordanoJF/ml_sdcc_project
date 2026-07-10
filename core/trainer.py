
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

    x, y = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y, label_smoothing=label_smoothing)
    loss.backward()
    if clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
    optimizer.step()
    return loss.item()


def compute_confusion_matrix(
    model: nn.Module,
    loader,
    device: torch.device,
) -> torch.Tensor:

    model.eval()
    confusion: torch.Tensor | None = None
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            num_classes = logits.shape[-1]
            if confusion is None:
                confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
            idx = (y * num_classes + preds).cpu()
            confusion += torch.bincount(idx, minlength=num_classes * num_classes).reshape(
                num_classes, num_classes
            )
    model.train()
    return confusion if confusion is not None else torch.zeros(0, 0, dtype=torch.long)


def macro_prf1_from_confusion(confusion: torch.Tensor) -> tuple[float, float, float]:

    if confusion.numel() == 0:
        return 0.0, 0.0, 0.0
    tp = confusion.diag().float()
    support = confusion.sum(dim=1).float()    # true count per class
    predicted = confusion.sum(dim=0).float()  # predicted count per class
    precision = torch.where(predicted > 0, tp / predicted.clamp(min=1), torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support.clamp(min=1), torch.zeros_like(tp))
    denom = (precision + recall).clamp(min=1e-12)
    f1 = torch.where((precision + recall) > 0, 2 * precision * recall / denom, torch.zeros_like(tp))
    present = support > 0
    if not present.any():
        return 0.0, 0.0, 0.0
    return precision[present].mean().item(), recall[present].mean().item(), f1[present].mean().item()


def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> tuple[float, float, float, float, float]:

    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    confusion: torch.Tensor | None = None
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += len(y)
            num_classes = logits.shape[-1]
            if confusion is None:
                confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
            idx = (y * num_classes + preds).cpu()
            confusion += torch.bincount(idx, minlength=num_classes * num_classes).reshape(
                num_classes, num_classes
            )
    model.train()
    if total == 0:
        return float("inf"), 0.0, 0.0, 0.0, 0.0

    macro_precision, macro_recall, macro_f1 = macro_prf1_from_confusion(confusion)
    return total_loss / total, correct / total, macro_precision, macro_recall, macro_f1
