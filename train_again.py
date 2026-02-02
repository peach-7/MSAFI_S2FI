# train_again.py
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
import glob

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


def list_available_weights(models_dir="trained_models"):
    """列出可用的权重文件"""
    os.makedirs(models_dir, exist_ok=True)
    weight_files = []

    # 搜索标准模型文件
    for model_name in CONFIG["all_model_names"]:
        standard_path = os.path.join(models_dir, f"{model_name}.pth")
        if os.path.exists(standard_path):
            weight_files.append(standard_path)

    # 搜索最佳模型文件
    best_models = glob.glob(os.path.join(models_dir, f"best_*_model.pth"))
    weight_files.extend(best_models)

    # 搜索所有pth文件
    all_pth = glob.glob(os.path.join(models_dir, "*.pth"))
    for pth in all_pth:
        if pth not in weight_files:
            weight_files.append(pth)

    return weight_files


def select_weights_file():
    """让用户选择权重文件"""
    print("\n" + "=" * 60)
    print("📂 选择预训练权重文件")
    print("=" * 60)

    weight_files = list_available_weights()

    if not weight_files:
        print("⚠️ 没有找到可用的权重文件")
        return None

    # 显示可用的权重文件
    for idx, file_path in enumerate(weight_files):
        # 获取相对路径以缩短显示
        relative_path = os.path.relpath(file_path, start=os.getcwd())
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转为MB
        print(f"{idx + 1}. {relative_path} ({file_size:.2f} MB)")

    print(f"{len(weight_files) + 1}. 不使用预训练权重（从头开始训练）")
    print(f"{len(weight_files) + 2}. 退出")
    print("=" * 60)

    while True:
        try:
            choice = input("\n请选择权重文件序号: ").strip()
            if not choice.isdigit():
                print("⚠️ 请输入数字序号")
                continue

            choice = int(choice)
            if choice == len(weight_files) + 2:
                print("👋 退出程序")
                sys.exit(0)
            elif choice == len(weight_files) + 1:
                print("\n✅ 选择从头开始训练")
                return None
            elif 1 <= choice <= len(weight_files):
                selected_file = weight_files[choice - 1]
                print(f"\n✅ 已选择权重文件: {os.path.relpath(selected_file, start=os.getcwd())}")
                return selected_file
            else:
                print(f"⚠️ 无效序号，请输入1-{len(weight_files) + 2}之间的数字")
        except KeyboardInterrupt:
            print("\n👋 程序中断")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️ 错误: {str(e)}")


def load_model_from_weights(model, weights_path, config):
    """从权重文件加载模型"""
    if weights_path is None:
        print("🔄 从头开始初始化模型")
        return model, None, 0

    try:
        print(f"📥 正在加载权重文件: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=config["device"])

        # 处理不同格式的检查点
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            optimizer_state = checkpoint.get('optimizer_state_dict', None)
            print(f"✅ 成功加载模型权重，起始轮次: {start_epoch}")
            return model, optimizer_state, start_epoch
        else:
            # 如果是直接保存的模型状态
            model.load_state_dict(checkpoint)
            print("✅ 成功加载模型权重，起始轮次: 0")
            return model, None, 0

    except Exception as e:
        print(f"❌ 加载权重失败: {str(e)}")
        print("🔄 将使用随机初始化的模型")
        return model, None, 0


def adjust_batch_size(epoch, initial_batch_size, batch_size_increase_factor, max_batch_size):
    new_batch_size = initial_batch_size * (batch_size_increase_factor ** epoch)
    return int(min(new_batch_size, max_batch_size))


def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, config, start_epoch=0):
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_f1_score = 0.0

    # 根据 model_name 动态构建 best_model_path
    best_model_path = os.path.join(os.path.dirname(config["model_paths"][model_name]), f"best_{model_name}_model.pth")
    patience = config["patience"]
    epochs_without_improvement = 0

    total_epochs = start_epoch + config["epochs"]

    for epoch in range(start_epoch, total_epochs):
        current_epoch = epoch - start_epoch + 1
        start_time = time.time()

        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 调整批量大小
        if epoch > start_epoch:  # 从第二轮开始调整
            new_batch_size = adjust_batch_size(
                current_epoch - 1,
                config["initial_batch_size"],
                config["batch_size_increase_factor"],
                config["max_batch_size"]
            )
            if new_batch_size != train_loader.batch_size:
                print(f"\n🔄 调整批量大小: {train_loader.batch_size} -> {new_batch_size}")
                train_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=new_batch_size,
                    shuffle=True,
                    num_workers=config["num_workers"],
                    pin_memory=True
                )

        print(f"\nEpoch {current_epoch}/{config['epochs']} | 当前批量大小: {train_loader.batch_size}")
        pbar = tqdm(train_loader, desc=f"训练 Epoch {current_epoch}/{config['epochs']}", ncols=120)

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
            val_pbar = tqdm(val_loader, desc=f"验证 Epoch {current_epoch}/{config['epochs']}", ncols=120)
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

        # Early Stopping 和保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            best_f1_score = f1_score_macro
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, best_model_path)
            print(f"💾 保存第{epoch + 1}轮训练模型: {best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"⚠️ Early Stopping: 连续 {patience} 个 epoch 验证集损失未下降，停止训练")
                break

        # 调用调度器
        scheduler.step(avg_val_loss)

        end_time = time.time()
        print(f"Epoch {current_epoch}/{config['epochs']} | 耗时: {end_time - start_time:.1f}s")

    return best_model_path


def main():
    print("=" * 80)
    print("🍎 继续训练水果分类模型")
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

    # 选择权重文件
    weights_file = select_weights_file()

    # 加载数据集
    print("\n[1/4] 加载数据集...")
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

    # 数据加载器（初始批量大小）
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["initial_batch_size"],
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
    print("\n[2/4] 加载模型架构...")
    model = get_model(model_name, CONFIG)
    model.to(CONFIG["device"])

    # 加载预训练权重
    print("\n[3/4] 加载预训练权重...")
    model, optimizer_state, start_epoch = load_model_from_weights(model, weights_file, CONFIG)

    # 损失函数和优化器
    print("\n[4/4] 设置优化器...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # 如果有保存的优化器状态，加载它
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("✅ 成功加载优化器状态")
        except Exception as e:
            print(f"⚠️ 加载优化器状态失败: {str(e)}，使用新的优化器")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 训练模型
    print(f"\n🚀 开始训练 (从第 {start_epoch + 1} 轮继续)...")
    best_model_path = train_model(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, CONFIG,
                                  start_epoch)

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"✅ 训练完成！总耗时: {total_time:.1f}秒")
    print(f"最佳模型路径: {best_model_path}")
    print(f"报告保存路径: {CONFIG['report_dir']}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()