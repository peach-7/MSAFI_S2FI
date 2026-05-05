# test.py
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score)
from tqdm import tqdm

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
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def count_model_parameters(model: torch.nn.Module) -> dict:
    """Compute parameter statistics for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total,
        'trainable_params': trainable,
        'non_trainable_params': total - trainable,
        'model_size_mb': total * 4 / 1024 / 1024,
    }


def print_model_parameters(model: torch.nn.Module, model_name: str) -> dict:
    info = count_model_parameters(model)
    print(f"\n[PARAMS] {model_name.upper()}")
    print(f"  Total:     {info['total_params']:,}")
    print(f"  Trainable: {info['trainable_params']:,}")
    print(f"  Size (MB): {info['model_size_mb']:.2f}")
    return info


def calculate_comprehensive_metrics(all_targets, all_preds, class_names):
    """Calculate full evaluation metrics."""
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
        'f1_macro': f1_macro, 'f1_weighted': f1_weighted, 'f1_micro': f1_micro,
        'precision_macro': precision_macro, 'recall_macro': recall_macro,
        'precision_weighted': precision_weighted, 'recall_weighted': recall_weighted,
        'class_accuracies': class_accuracies,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(all_targets, all_preds),
        'class_names': class_names,
    }


def print_comprehensive_metrics(metrics: dict, model_name: str):
    """Print formatted evaluation results."""
    print(f"\n{'=' * 70}")
    print(f"[EVAL] {model_name.upper()}")
    print(f"{'=' * 70}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Macro F1:       {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1:    {metrics['f1_weighted']:.4f}")
    print(f"  Macro Precision:{metrics['precision_macro']:.4f}")
    print(f"  Macro Recall:   {metrics['recall_macro']:.4f}")

    report = metrics['classification_report']
    for cls in metrics['class_names']:
        if cls in report:
            m = report[cls]
            print(f"  [{cls}] F1={m['f1-score']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  Acc={metrics['class_accuracies'][cls]:.4f}")

    if 'macro avg' in report:
        ma = report['macro avg']
        print(f"  [macro avg] F1={ma['f1-score']:.4f}  P={ma['precision']:.4f}  R={ma['recall']:.4f}")
    if 'weighted avg' in report:
        wa = report['weighted avg']
        print(f"  [weighted avg] F1={wa['f1-score']:.4f}  P={wa['precision']:.4f}  R={wa['recall']:.4f}")
    print(f"{'=' * 70}")


def get_model(model_name: str, config: dict) -> torch.nn.Module:
    """Instantiate a model by name."""
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


def load_best_model(model_name: str, config: dict) -> torch.nn.Module:
    """Load model weights from checkpoint."""
    model = get_model(model_name, config)
    model_path = config["model_paths"].get(model_name)
    if model_path is None:
        model_path = os.path.join("trained_models", f"best_{model_name}_model.pth")

    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=config["device"], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model: {model_path}")
    return model


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, config: dict) -> dict:
    """Run inference on test set and compute metrics."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Evaluating"):
            data = data.to(config["device"])
            targets = targets.to(config["device"])
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    metrics = calculate_comprehensive_metrics(all_targets, all_preds, config['class_names'])
    metrics['predictions'] = all_preds
    metrics['targets'] = all_targets
    return metrics


def save_results(metrics: dict, model_name: str, config: dict):
    """Save classification report, confusion matrix, and plots."""
    report_dir = config["report_dir"]
    os.makedirs(report_dir, exist_ok=True)

    # Classification report CSV
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    report_df.to_csv(os.path.join(report_dir, f"{model_name}_classification_report.csv"),
                     encoding='utf-8-sig')

    # Confusion matrix CSV
    cm_df = pd.DataFrame(metrics['confusion_matrix'],
                         index=config['class_names'], columns=config['class_names'])
    cm_df.to_csv(os.path.join(report_dir, f"{model_name}_confusion_matrix.csv"),
                 encoding='utf-8-sig')

    # Confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=config['class_names'], yticklabels=config['class_names'])
    plt.title(f'{model_name.upper()} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.close()

    # Per-class performance bar chart
    metrics_list = ['precision', 'recall', 'f1-score']
    names = config['class_names'].copy()
    data_plot = {m: [metrics['classification_report'][c][m] for c in names] for m in metrics_list}
    names.append('macro avg')
    for m in metrics_list:
        data_plot[m].append(metrics['classification_report']['macro avg'][m])

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, m in enumerate(metrics_list):
        bars = ax.bar(x + (i - 1) * width, data_plot[m], width, label=m, alpha=0.8)
        for bar in bars:
            ax.annotate(f'{bar.get_height():.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name.upper()} Classification Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"{model_name}_performance_report.png"), dpi=300)
    plt.close()

    # Text summary
    with open(os.path.join(report_dir, f"{model_name}_metrics.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Model Evaluation Report: {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy:        {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1:        {metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted F1:     {metrics['f1_weighted']:.4f}\n")
        f.write(f"Macro Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"Macro Recall:    {metrics['recall_macro']:.4f}\n\n")
        for cls in metrics['class_names']:
            if cls in metrics['classification_report']:
                m = metrics['classification_report'][cls]
                f.write(f"[{cls}] F1={m['f1-score']:.4f}  P={m['precision']:.4f}  "
                        f"R={m['recall']:.4f}  Acc={metrics['class_accuracies'][cls]:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")

    print(f"Results saved to {report_dir}/")


def main():
    print("=" * 70)
    print("Fruit Classification Testing")
    print("=" * 70)
    config = CONFIG
    print(f"Test data: {config['test_data_dir']}")
    print(f"Device: {config['device'].upper()}")

    # Model selection
    print("\nAvailable models:")
    for idx, name in enumerate(config["all_model_names"]):
        print(f"  {idx + 1}. {name}")
    print(f"  {len(config['all_model_names']) + 1}. Exit")

    while True:
        try:
            choice = input("\nSelect model number: ").strip()
            if not choice.isdigit():
                continue
            choice = int(choice)
            if choice == len(config["all_model_names"]) + 1:
                sys.exit(0)
            if 1 <= choice <= len(config["all_model_names"]):
                model_name = config["all_model_names"][choice - 1]
                print(f"Selected: {model_name}")
                break
        except (KeyboardInterrupt, ValueError):
            sys.exit(0)

    # Load test data
    print("\n[1/3] Loading test dataset...")
    test_dataset = FruitVideoDataset(
        data_dir=config["test_data_dir"], model_name=model_name,
        num_frames=config["num_frames"], input_size=config["input_size"], train=False
    )
    if len(test_dataset) == 0:
        print("Error: empty test dataset.")
        sys.exit(1)

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=config["num_workers"], pin_memory=True)

    # Load model
    print("\n[2/3] Loading model...")
    model = load_best_model(model_name, config)
    model.to(config["device"])
    print_model_parameters(model, model_name)

    # Evaluate
    print("\n[3/3] Evaluating...")
    metrics = evaluate_model(model, test_loader, config)
    save_results(metrics, model_name, config)
    print_comprehensive_metrics(metrics, model_name)

    info = count_model_parameters(model)
    print(f"\nModel: {model_name} | Params: {info['total_params']:,} | Size: {info['model_size_mb']:.2f} MB")
    print("Done.")


if __name__ == "__main__":
    main()