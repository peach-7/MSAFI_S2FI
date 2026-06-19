# train.py
import os
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix)

from config import CONFIG
from utils.data_processing import FruitVideoDataset

from models.C3D import C3D
from models.I3D import I3D
from models.R3D import R3D
from models.MC3D import MC3D
from models.Transformer import Transformer
from models.DVFLNet import DVFLNet
from models.MSAFI import MSAFI
from models.mamba import MambaVideo
from models.MoCLIP_Lite import MoCLIP_Lite
from models.MoViNets import MoViNets

warnings.filterwarnings('ignore')


def count_model_parameters(model: nn.Module) -> dict:
    """Compute total, trainable, and non-trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total,
        'trainable_params': trainable,
        'non_trainable_params': total - trainable,
        'model_size_mb': total * 4 / 1024 / 1024,
    }


def print_model_parameters(model: nn.Module, model_name: str) -> dict:
    """Print and return parameter statistics for a model."""
    info = count_model_parameters(model)
    print(f"\n[PARAMS] {model_name.upper()}")
    print(f"  Total:         {info['total_params']:,}")
    print(f"  Trainable:     {info['trainable_params']:,}")
    print(f"  Non-trainable: {info['non_trainable_params']:,}")
    print(f"  Size (MB):     {info['model_size_mb']:.2f}")
    return info


def calculate_comprehensive_metrics(all_targets, all_preds, class_names):
    """Calculate accuracy, F1, precision, recall, per-class metrics, and confusion matrix."""
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    accuracy = np.mean(all_targets == all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    f1_weighted = f1_score(all_targets, all_preds, average='weighted')
    f1_micro = f1_score(all_targets, all_preds, average='micro')
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    precision_weighted = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_targets, all_preds, average='weighted', zero_division=0)

    report = classification_report(all_targets, all_preds, target_names=class_names,
                                   output_dict=True, zero_division=0)

    class_accuracies = {}
    for i, name in enumerate(class_names):
        mask = (all_targets == i)
        if mask.sum() > 0:
            class_accuracies[name] = np.sum(all_preds[mask] == all_targets[mask]) / mask.sum()
        else:
            class_accuracies[name] = 0.0

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'class_accuracies': class_accuracies,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(all_targets, all_preds),
        'class_names': class_names,
    }


def print_comprehensive_metrics(metrics: dict, model_name: str, epoch: int = None):
    """Print formatted evaluation metrics to stdout."""
    header = f"[EVAL] {model_name.upper()}"
    if epoch is not None:
        header += f" - Epoch {epoch}"
    print(f"\n{'=' * 70}")
    print(header)
    print(f"{'=' * 70}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro F1:        {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1:     {metrics['f1_weighted']:.4f}")
    print(f"  Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"  Macro Recall:    {metrics['recall_macro']:.4f}")

    report = metrics['classification_report']
    for cls in metrics['class_names']:
        if cls in report:
            m = report[cls]
            print(f"  [{cls}] F1={m['f1-score']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  Acc={metrics['class_accuracies'][cls]:.4f}")
    print(f"{'=' * 70}")


class SuperAdaptiveCompositeLoss(nn.Module):
    """SACL: Super Adaptive Composite Loss for imbalanced datasets.

    Combines focal loss with adaptive class weighting and label smoothing.
    All key hyper-parameters are learnable and adjusted during training.

    Args:
        num_classes: Number of target classes.
        device: Target device string.
        initial_gamma: Starting value for the focal loss exponent.
        initial_alpha: Starting value for class-weight blending factor.
        class_weights: Optional manual class weight list.
    """

    def __init__(self, num_classes: int = 3, device: str = 'cuda',
                 initial_gamma: float = 2.0, initial_alpha: float = 1.0,
                 class_weights: list = None):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

        if class_weights is not None:
            self.register_buffer('manual_class_weights',
                                 torch.tensor(class_weights, device=device, dtype=torch.float32))
        else:
            total = 3683 + 4574 + 131
            w = [total / (3 * 3683), total / (3 * 4574), total / (3 * 131)]
            mean_w = sum(w) / 3
            w = [x / mean_w for x in w]
            self.register_buffer('manual_class_weights',
                                 torch.tensor(w, device=device, dtype=torch.float32))

        self.gamma = nn.Parameter(torch.tensor(initial_gamma))
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        self.label_smoothing = nn.Parameter(torch.tensor(0.1))
        self.focal_weight = nn.Parameter(torch.tensor(0.7))
        self.smooth_weight = nn.Parameter(torch.tensor(0.3))

        self.register_buffer('class_counts', torch.ones(num_classes, device=device))
        self.register_buffer('class_losses', torch.zeros(num_classes, device=device))
        self.register_buffer('sample_difficulties', torch.ones(10000, device=device))
        self.register_buffer('running_loss_mean', torch.tensor(0.0, device=device))
        self.register_buffer('running_loss_std', torch.tensor(1.0, device=device))
        self._sample_idx = 0

    def update_statistics(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Update online class-frequency and difficulty statistics."""
        with torch.no_grad():
            targets = targets.to(self.device)
            probs = F.softmax(inputs, dim=1)
            correct_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            for cls in range(self.num_classes):
                mask = (targets == cls)
                if mask.any():
                    self.class_losses[cls] = self.class_losses[cls] * 0.9 + correct_probs[mask].mean() * 0.1
                    self.class_counts[cls] += mask.sum()
            difficulties = 1.0 - correct_probs
            for d in difficulties:
                if self._sample_idx < len(self.sample_difficulties):
                    self.sample_difficulties[self._sample_idx] = d
                    self._sample_idx += 1
                else:
                    self._sample_idx = 0

    def adaptive_focal_factor(self, probs_gt: torch.Tensor) -> torch.Tensor:
        """Compute difficulty-aware focal scaling factor."""
        gamma = torch.clamp(self.gamma, 0.5, 5.0)
        focal = (1.0 - probs_gt) ** gamma
        if self._sample_idx > 0:
            mean_d = self.sample_difficulties[:self._sample_idx].mean()
        else:
            mean_d = torch.tensor(0.5, device=self.device)
        adj = 1.0 + 0.5 * torch.tanh(mean_d - 0.5)
        return focal * adj

    def adaptive_class_weights(self) -> torch.Tensor:
        """Compute class weights blending frequency, performance, and manual priors."""
        freq_w = 1.0 / torch.sqrt(self.class_counts)
        perf_w = 1.0 / (self.class_losses + 1e-8)
        weights = freq_w * perf_w
        weights = weights / weights.mean()
        factor = torch.sigmoid(self.alpha)
        combined = (1 - factor) * weights + factor * self.manual_class_weights
        return torch.clamp(combined, 0.1, 50.0)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the composite loss value.

        Args:
            inputs: Raw logits of shape (N, C).
            targets: Ground-truth class indices of shape (N,).

        Returns:
            Scalar loss tensor.
        """
        targets = targets.to(self.device)
        self.update_statistics(inputs, targets)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        probs_gt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = self.adaptive_focal_factor(probs_gt)
        focal_loss = focal_factor * ce_loss
        class_w = self.adaptive_class_weights()
        weighted_focal = class_w[targets] * focal_loss

        ls = torch.clamp(self.label_smoothing, 0.01, 0.3)
        smooth_labels = torch.full_like(inputs, ls.item() / (self.num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - ls.item())
        smooth_ce = -(smooth_labels * log_probs).sum(dim=1)

        fw = torch.sigmoid(self.focal_weight)
        sw = torch.sigmoid(self.smooth_weight)
        combined = fw * weighted_focal + sw * smooth_ce

        with torch.no_grad():
            batch_mean = combined.mean()
            batch_std = torch.sqrt(((combined - batch_mean) ** 2).mean() + 1e-8)
            self.running_loss_mean = 0.9 * self.running_loss_mean + 0.1 * batch_mean
            self.running_loss_std = 0.9 * self.running_loss_std + 0.1 * batch_std

        standardized = (combined - self.running_loss_mean) / (self.running_loss_std + 1e-8)
        grad_weights = torch.exp(torch.clamp(standardized, -2, 2) * 0.5)
        return (grad_weights * combined).mean()


def get_model(model_name: str, config: dict) -> nn.Module:
    """Instantiate a model by name from the configuration registry.

    Args:
        model_name: One of the keys in config['all_model_names'].
        config: Global configuration dictionary.

    Returns:
        Instantiated nn.Module on CPU.

    Raises:
        ValueError: If model_name is not registered.
    """
    nc = config["num_classes"]
    nf = config["num_frames"]
    isz = config["input_size"]

    registry = {
        "r3d": lambda: R3D(num_classes=nc),
        "i3d": lambda: I3D(num_classes=nc),
        "c3d": lambda: C3D(num_classes=nc, num_frames=nf, input_size=isz),
        "mc3d": lambda: MC3D(num_classes=nc, num_frames=nf, input_size=isz),
        "transformer": lambda: Transformer(num_classes=nc, num_frames=nf, input_size=isz),
        "mamba": lambda: MambaVideo(num_classes=nc, num_frames=nf, input_size=isz),
        "DVFLNet": lambda: DVFLNet(num_classes=nc, num_frames=nf, img_size=isz),
        "msafi": lambda: MSAFI(num_classes=nc, num_frames=nf, img_size=isz),
        "moclip_lite": lambda: MoCLIP_Lite(num_classes=nc, num_frames=nf, input_size=isz),
        "movinets": lambda: MoViNets(num_classes=nc, num_frames=nf, input_size=isz),
    }

    if model_name not in registry:
        raise ValueError(f"Unsupported model: {model_name}")
    return registry[model_name]()


def train_model(model: nn.Module, model_name: str, train_loader: DataLoader,
                val_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler, config: dict) -> str:
    """Train model with mixed-precision and gradient accumulation.

    Returns:
        File path of the saved best-model checkpoint.
    """
    best_f1 = 0.0
    patience = config["patience"]
    no_improve = 0
    model_dir = os.path.dirname(config["model_paths"][model_name])
    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, f"best_{model_name}_model.pth")

    use_amp = config.get("use_amp", True)
    scaler = GradScaler(enabled=use_amp)
    grad_accum = config.get("gradient_accumulation_steps", 1)

    print(f"[TRAIN] AMP={'ON' if use_amp else 'OFF'}, GradAccum={grad_accum}, "
          f"EffectiveBatch={train_loader.batch_size * grad_accum}")

    for epoch in range(config["epochs"]):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Train {epoch + 1}/{config['epochs']}", ncols=100)
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(config["device"])
            targets = targets.to(config["device"])

            with autocast(enabled=use_amp):
                outputs = model(data)
                loss = criterion(outputs, targets)
                if grad_accum > 1:
                    loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(criterion.parameters()), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum * data.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == targets).sum().item()
            total_samples += data.size(0)

            pbar.set_postfix({
                'loss': f"{total_loss / total_samples:.4f}",
                'acc': f"{total_correct / total_samples:.4f}"
            })

        if (batch_idx + 1) % grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_loss, val_correct, val_samples = 0.0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc=f"Val {epoch + 1}/{config['epochs']}", ncols=100):
                data = data.to(config["device"])
                targets = targets.to(config["device"])
                with autocast(enabled=use_amp):
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * data.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == targets).sum().item()
                val_samples += data.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        metrics = calculate_comprehensive_metrics(all_targets, all_preds, config['class_names'])
        print_comprehensive_metrics(metrics, model_name, epoch + 1)

        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if use_amp else None,
                'f1_score': metrics['f1_macro'],
                'metrics': metrics,
            }, best_path)
            print(f"[SAVE] Epoch {epoch + 1} -> {best_path} (Macro F1: {best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EARLY STOP] No improvement for {patience} epochs.")
                break

        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch + 1} done in {elapsed:.1f}s")

        if config["device"] == "cuda":
            mem = torch.cuda.memory_allocated() / 1024 ** 3
            print(f"[GPU] Memory allocated: {mem:.2f} GB")

    return best_path


def main():
    """Entry point for interactive model training."""
    print("=" * 70)
    print("Fruit Classification Training")
    print("=" * 70)
    print(f"Data: {CONFIG['train_data_dir']}")
    print(f"Device: {CONFIG['device'].upper()}")
    print(f"Frames: {CONFIG['num_frames']}, Input size: {CONFIG['input_size']}")

    os.makedirs(CONFIG["report_dir"], exist_ok=True)

    print("\nAvailable models:")
    for idx, name in enumerate(CONFIG["all_model_names"]):
        print(f"  {idx + 1}. {name}")
    print(f"  {len(CONFIG['all_model_names']) + 1}. Exit")

    while True:
        try:
            choice = input("\nSelect model number: ").strip()
            if not choice.isdigit():
                continue
            choice = int(choice)
            if choice == len(CONFIG["all_model_names"]) + 1:
                sys.exit(0)
            if 1 <= choice <= len(CONFIG["all_model_names"]):
                model_name = CONFIG["all_model_names"][choice - 1]
                print(f"Selected: {model_name}")
                break
        except (KeyboardInterrupt, ValueError):
            sys.exit(0)

    # Load datasets
    print("\n[1/3] Loading datasets...")
    train_dataset = FruitVideoDataset(
        data_dir=CONFIG["train_data_dir"],
        model_name=model_name,
        num_frames=CONFIG["num_frames"],
        input_size=CONFIG["input_size"],
        train=True,
        augment=True,
        target_count=CONFIG["target_count"]
    )

    train_sampler = None
    if hasattr(train_dataset, 'get_weighted_sampler'):
        train_sampler = train_dataset.get_weighted_sampler()
    val_dataset = FruitVideoDataset(
        data_dir=CONFIG["val_data_dir"],
        model_name=model_name,
        num_frames=CONFIG["num_frames"],
        input_size=CONFIG["input_size"],
        train=False
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: empty dataset.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["initial_batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    # Build model
    print("\n[2/3] Building model...")
    model = get_model(model_name, CONFIG)
    model.to(CONFIG["device"])
    print_model_parameters(model, model_name)

    # Loss, optimizer, scheduler
    criterion = SuperAdaptiveCompositeLoss(
        num_classes=CONFIG["num_classes"],
        device=CONFIG["device"],
        initial_gamma=2.0,
        initial_alpha=1.0,
        class_weights=CONFIG["class_weights"]
    ).to(CONFIG["device"])

    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )

    # Train
    print("\n[3/3] Training...")
    t0 = time.time()
    best_path = train_model(
        model, model_name, train_loader, val_loader,
        criterion, optimizer, scheduler, CONFIG
    )

    elapsed = time.time() - t0
    info = count_model_parameters(model)
    print(f"\n{'=' * 70}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Model: {model_name} | Params: {info['total_params']:,} | Size: {info['model_size_mb']:.2f} MB")
    print(f"Best model: {best_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
