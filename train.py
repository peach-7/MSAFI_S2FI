# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_processing import FruitVideoDataset
from models.C3D import C3D
from models.I3D import I3D
from models.R3D import R3D
from models.MC3D import MC3D
from models.Transformer import Transformer
from models.Mamba_official import Mamba_official
import warnings
import os
import time
import sys

warnings.filterwarnings('ignore')

# 加载配置
from config import CONFIG


def get_model(model_name, config):
    if model_name == "r3d":
        return R3D(num_classes=config["num_classes"])
    elif model_name == "i3d":
        return I3D(num_classes=config["num_classes"])
    elif model_name == "c3d":
        return C3D(num_classes=config["num_classes"], num_frames=config["num_frames"], input_size=config["input_size"])
    elif model_name == "mc3d":
        return MC3D(num_classes=config["num_classes"], num_frames=config["num_frames"], input_size=config["input_size"])
    elif model_name == "transformer":
        return Transformer(num_classes=config["num_classes"], num_frames=config["num_frames"],
                           input_size=config["input_size"])
    elif model_name == "mamba_official":
        return Mamba_official(num_classes=config["num_classes"], num_frames=config["num_frames"],
                              input_size=config["input_size"])
    else:
        raise ValueError(f"不支持的模型：{model_name}")


def adjust_batch_size(epoch, initial_batch_size, batch_size_increase_factor, max_batch_size):
    new_batch_size = initial_batch_size * (batch_size_increase_factor ** epoch)
    return int(min(new_batch_size, max_batch_size))


def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, config):
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_f1_score = 0.0  # 用于保存最佳模型的关键指标

    # 根据 model_name 动态构建 best_model_path
    best_model_path = os.path.join(os.path.dirname(config["model_paths"][model_name]), f"best_{model_name}_model.pth")
    patience = config["patience"]
    epochs_without_improvement = 0

    for epoch in range(config["epochs"]):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        print(f"\nEpoch {epoch + 1}/{config['epochs']} | 当前批量大小: {train_loader.batch_size}")
        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{config['epochs']}", ncols=120)
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(config["device"])
            targets = targets.to(config["device"])

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == targets).sum().item()
            total_samples += data.size(0)

            pbar.set_postfix({
                '平均损失': f"{total_loss / total_samples:.4f}",
                '准确率': f"{total_correct / total_samples:.4f}"
            })

        avg_train_loss = total_loss / total_samples
        avg_train_acc = total_correct / total_samples

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        class_correct = [0] * config["num_classes"]
        class_total = [0] * config["num_classes"]

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"验证 Epoch {epoch + 1}/{config['epochs']}", ncols=120)
            for batch_idx, (data, targets) in enumerate(val_pbar):
                data = data.to(config["device"])
                targets = targets.to(config["device"])

                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * data.size(0)

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == targets).sum().item()
                val_samples += data.size(0)

                # 计算每个类别的准确率
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    class_correct[t] += p.eq(t).item()
                    class_total[t] += 1

            val_pbar.set_postfix({
                '平均损失': f"{val_loss / val_samples:.4f}",
                '准确率': f"{val_correct / val_samples:.4f}"
            })

        avg_val_loss = val_loss / val_samples
        avg_val_acc = val_correct / val_samples
        class_accuracies = [class_correct[i] / class_total[i] if class_total[i] != 0 else 0 for i in
                            range(config["num_classes"])]
        avg_class_accuracy = sum(class_accuracies) / len(class_accuracies)

        # 计算F1分数
        from sklearn.metrics import f1_score
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                data = data.to(config["device"])
                targets = targets.to(config["device"])
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        f1_score_macro = f1_score(all_targets, all_preds, average='macro')

        print(f"验证 | 损失: {avg_val_loss:.4f} | 准确率: {avg_val_acc:.4f} | F1: {f1_score_macro:.4f}")
        print(
            f"验证类别准确率 | {config['class_names'][0]}: {class_accuracies[0]:.4f} | {config['class_names'][1]}: {class_accuracies[1]:.4f} | {config['class_names'][2]}: {class_accuracies[2]:.4f} | 平均: {avg_class_accuracy:.4f}")
        print(f"最佳 | 准确率: {best_val_acc:.4f} | F1: {best_f1_score:.4f}")
        print(f"学习率: {scheduler.optimizer.param_groups[0]['lr']}")

        # ===== 修改1：使用Macro F1作为唯一保存指标 =====
        if f1_score_macro > best_f1_score:  # 注意：F1分数是越大越好
            best_f1_score = f1_score_macro
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': f1_score_macro,
            }, best_model_path)
            print(f"💾 保存第{epoch + 1}轮训练模型: {best_model_path} (Macro F1: {f1_score_macro:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"⚠️ Early Stopping: 连续 {patience} 个 epoch Macro F1 未提升，停止训练")
                break

        # ===== 修改2：调度器基于Macro F1 (mode='max') =====
        scheduler.step(f1_score_macro)  # 注意：这里传入的是F1分数，而不是损失

        end_time = time.time()
        print(f"Epoch {epoch + 1}/{config['epochs']} | 耗时: {end_time - start_time:.1f}s")

    return best_model_path


def main():
    print("=" * 80)
    print("🍎 训练水果分类模型")
    print("=" * 80)
    print(f"📁 数据集: {CONFIG['train_data_dir']}")
    print(f"🔧 设备: {CONFIG['device'].upper()}")
    print(f"📊 帧数: {CONFIG['num_frames']} | 图像大小: {CONFIG['input_size']}")

    start_time = time.time()

    # 创建报告目录
    os.makedirs(CONFIG["report_dir"], exist_ok=True)

    # 选择模型
    print("\n" + "=" * 60)
    print("📌 模型选择界面")
    print("=" * 60)
    for idx, name in enumerate(CONFIG["all_model_names"]):
        print(f"{idx + 1}. {name.upper()}")
    print(f"{len(CONFIG['all_model_names']) + 1}. 退出")
    print("=" * 60)

    while True:
        try:
            choice = input("\n请选择模型序号: ").strip()
            if not choice.isdigit():
                print("⚠️ 请输入数字序号")
                continue

            choice = int(choice)
            if choice == len(CONFIG["all_model_names"]) + 1:
                print("👋 退出程序")
                sys.exit(0)
            elif 1 <= choice <= len(CONFIG["all_model_names"]):
                model_name = CONFIG["all_model_names"][choice - 1]
                print(f"\n✅ 已选择: {model_name.upper()}")
                break
            else:
                print(f"⚠️ 无效序号，请输入1-{len(CONFIG['all_model_names']) + 1}之间的数字")
        except KeyboardInterrupt:
            print("\n👋 程序中断")
            sys.exit(0)
        except ValueError:
            print("⚠️ 请输入有效数字")

    # 加载数据集
    print("\n[1/3] 加载数据集...")
    train_dataset = FruitVideoDataset(
        data_dir=CONFIG["train_data_dir"],
        model_name=model_name,
        num_frames=CONFIG["num_frames"],
        input_size=CONFIG["input_size"],
        train=True,
        augment=True,  # 启用数据增强
        target_count=CONFIG["target_count"]  # 使用配置中的目标数量
    )

    val_dataset = FruitVideoDataset(
        data_dir=CONFIG["val_data_dir"],
        model_name=model_name,
        num_frames=CONFIG["num_frames"],
        input_size=CONFIG["input_size"],
        train=False
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ 错误：数据集为空，请检查路径")
        sys.exit(1)

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["initial_batch_size"],  # 初始批量大小
        shuffle=True,
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

    # 加载模型
    print("\n[2/3] 加载模型...")
    model = get_model(model_name, CONFIG)
    model.to(CONFIG["device"])

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # ===== 修改3：调度器改为mode='max'以监控F1分数 =====
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控最大化指标（F1分数）
        factor=0.1,
        patience=5
    )

    # 训练模型
    print("\n[3/3] 开始训练...")
    best_model_path = train_model(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, CONFIG)

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"✅ 训练完成！总耗时: {total_time:.1f}秒")
    print(f"最佳模型路径: {best_model_path}")
    print(f"报告保存路径: {CONFIG['report_dir']}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()