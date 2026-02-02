import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from utils.data_processing import FruitVideoDataset
from models.C3D import C3D
from models.I3D import I3D
from models.R3D import R3D
from models.MC3D import MC3D
from models.Transformer import Transformer
from models.Mamba_official import Mamba_official
import warnings
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


def load_best_model(model_name, config):
    model = get_model(model_name, config)
    checkpoint_path = config["model_paths"][model_name]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config["device"],weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config["device"])
    model.eval()
    print(f"✅ 已加载最佳模型: {checkpoint_path}")
    return model


def test_model(model, test_loader, config):
    model.eval()
    total_correct = 0
    total_samples = 0
    test_preds = []
    test_targets = []
    error_samples = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="测试", ncols=120)
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(config["device"])
            targets = targets.to(config["device"])
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == targets).sum().item()
            total_samples += data.size(0)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

            # 记录错误样本
            for i in range(data.size(0)):
                if preds[i] != targets[i]:
                    video_path = test_loader.dataset.video_paths[batch_idx * config["batch_size"] + i]
                    error_samples.append({
                        "video_path": video_path,
                        "true_label": targets[i].item(),
                        "predicted_label": preds[i].item()
                    })

            pbar.set_postfix({
                '准确率': f"{total_correct / total_samples:.4f}"
            })

    avg_acc = total_correct / total_samples
    return avg_acc, test_preds, test_targets, error_samples


def generate_classification_report(y_true, y_pred, class_names, report_dir, model_name):
    # 确保使用中文类别名称
    chinese_class_names = ["好果", "次果", "烂果"]

    # 生成分类报告
    report = classification_report(y_true, y_pred, target_names=chinese_class_names, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # 保存分类报告
    report_path = os.path.join(report_dir, f"{model_name}_classification_report.csv")
    report_df.to_csv(report_path, index=True)
    print(f"✅ 分类报告已保存: {report_path}")

    # 提取宏平均F1和平均准确率
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']
    accuracy = report['accuracy']

    # 打印详细中文结果
    print("\n" + "=" * 65)
    print(f"📊 {model_name.upper()} 详细测试结果 (中文版)")
    print("=" * 65)
    for i, class_name in enumerate(chinese_class_names):
        class_report = report[class_name]
        print(f"{class_name} - 准确率: {class_report['precision']:.4f}, "
              f"召回率: {class_report['recall']:.4f}, "
              f"F1分数: {class_report['f1-score']:.4f}, "
              f"样本数: {int(class_report['support'])}")

    print("-" * 65)
    print(f"整体准确率: {accuracy:.4f}")
    print(f"平均准确率 (宏平均): {macro_avg['precision']:.4f}")
    print(f"宏平均F1: {macro_avg['f1-score']:.4f}")
    print("=" * 65)

    # 保存中文格式的测试结果
    chinese_report_path = os.path.join(report_dir, f"{model_name}_中文测试结果.txt")
    with open(chinese_report_path, 'w', encoding='utf-8') as f:
        f.write(f"📊 {model_name.upper()} 详细测试结果 (中文版)\n")
        f.write("=" * 65 + "\n")
        for i, class_name in enumerate(chinese_class_names):
            class_report = report[class_name]
            f.write(f"{class_name} - 准确率: {class_report['precision']:.4f}, "
                    f"召回率: {class_report['recall']:.4f}, "
                    f"F1分数: {class_report['f1-score']:.4f}, "
                    f"样本数: {int(class_report['support'])}\n")

        f.write("-" * 65 + "\n")
        f.write(f"整体准确率: {accuracy:.4f}\n")
        f.write(f"平均准确率 (宏平均): {macro_avg['precision']:.4f}\n")
        f.write(f"宏平均F1: {macro_avg['f1-score']:.4f}\n")
        f.write("=" * 65 + "\n")

    print(f"✅ 中文测试结果已保存: {chinese_report_path}")
    return report_df, macro_avg['precision'], macro_avg['f1-score'], accuracy


def plot_confusion_matrix(y_true, y_pred, class_names, report_dir, model_name):
    # 使用中文类别名称
    chinese_class_names = ["好果", "次果", "烂果"]

    cm = confusion_matrix(y_true, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=chinese_class_names)
    plt.figure(figsize=(10, 8))
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name.upper()} 混淆矩阵")
    plt.ylabel("真实标签")
    plt.xlabel("预测标签")
    plt.grid(False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    cm_path = os.path.join(report_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ 混淆矩阵图已保存: {cm_path}")
    return cm


def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, report_dir, model_name):
    # 使用中文类别名称
    chinese_class_names = ["好果", "次果", "烂果"]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=chinese_class_names, yticklabels=chinese_class_names)
    plt.title(f"{model_name.upper()} 混淆矩阵热力图")
    plt.ylabel("真实标签")
    plt.xlabel("预测标签")
    plt.xticks(rotation=45)
    plt.tight_layout()
    cm_heatmap_path = os.path.join(report_dir, f"{model_name}_confusion_matrix_heatmap.png")
    plt.savefig(cm_heatmap_path)
    plt.close()
    print(f"✅ 混淆矩阵热力图已保存: {cm_heatmap_path}")
    return cm


def save_error_samples(error_samples, report_dir, model_name):
    error_df = pd.DataFrame(error_samples)
    error_path = os.path.join(report_dir, f"{model_name}_error_samples.csv")
    error_df.to_csv(error_path, index=False)
    print(f"✅ 错误样本已保存: {error_path}")


def main():
    print("=" * 80)
    print("🍎 水果分类模型测试工具（增强版）")
    print("=" * 80)
    print(f"📁 测试集: {CONFIG['test_data_dir']}")
    print(f"🔧 设备: {CONFIG['device'].upper()}")
    print(f"📦 权重目录: trained_models")
    start_time = time.time()

    # 创建报告目录
    os.makedirs(CONFIG["report_dir"], exist_ok=True)

    # 选择模型
    print("\n" + "=" * 60)
    print("📌 模型选择界面")
    print("=" * 60)
    for idx, name in enumerate(CONFIG["all_model_names"]):
        print(f"{idx + 1}. {name.upper()} - {'✅ 存在' if os.path.exists(CONFIG['model_paths'][name]) else '❌ 缺失'}")
    print(f"{len(CONFIG['all_model_names']) + 1}. 测试所有模型")
    print(f"{len(CONFIG['all_model_names']) + 2}. 退出")
    print("=" * 60)

    while True:
        try:
            choice = input("\n请输入序号（如1,3）：").strip()
            if not choice.isdigit():
                print("⚠️ 请输入数字序号")
                continue
            choice = int(choice)
            if choice == len(CONFIG["all_model_names"]) + 2:
                print("👋 退出程序")
                sys.exit(0)
            elif choice == len(CONFIG["all_model_names"]) + 1:
                print(f"\n✅ 已选择: 所有模型")
                selected_models = CONFIG["all_model_names"]
                break
            elif 1 <= choice <= len(CONFIG["all_model_names"]):
                model_name = CONFIG["all_model_names"][choice - 1]
                print(f"\n✅ 已选择: {model_name.upper()}")
                selected_models = [model_name]
                break
            else:
                print(f"⚠️ 无效序号，请输入1-{len(CONFIG['all_model_names']) + 2}之间的数字")
        except KeyboardInterrupt:
            print("\n👋 程序中断")
            sys.exit(0)
        except ValueError:
            print("⚠️ 请输入有效数字")

    # 加载数据集
    print("\n📊 加载测试集...")
    # 使用第一个模型的名称获取数据集（所有模型使用相同的数据集）
    model_name_for_dataset = selected_models[0]
    test_dataset = FruitVideoDataset(
        data_dir=CONFIG["test_data_dir"],
        model_name=model_name_for_dataset,
        num_frames=CONFIG["num_frames"],
        input_size=CONFIG["input_size"],
        train=False
    )
    if len(test_dataset) == 0:
        print("❌ 错误：数据集为空，请检查路径")
        sys.exit(1)

    # 数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    # 总测试结果
    overall_results = {}

    for model_name in selected_models:
        print(f"\n{'=' * 83}")
        print(f"[1/1] 测试模型: {model_name.upper()}")
        print(f"{'=' * 83}")

        # 加载模型
        try:
            model = load_best_model(model_name, CONFIG)
        except FileNotFoundError:
            print(f"⚠️ 模型文件不存在: {CONFIG['model_paths'][model_name]}")
            continue

        # 测试模型
        print("\n🚀 开始推理测试...")
        test_acc, test_preds, test_targets, error_samples = test_model(model, test_loader, CONFIG)

        # 生成分类报告（使用中文显示）
        print("\n📊 生成分类报告...")
        report_df, macro_precision, macro_f1, accuracy = generate_classification_report(
            test_targets, test_preds, CONFIG["class_names"], CONFIG["report_dir"], model_name
        )

        # 绘制混淆矩阵
        print("\n📊 绘制混淆矩阵...")
        cm = plot_confusion_matrix(test_targets, test_preds, CONFIG["class_names"], CONFIG["report_dir"], model_name)
        cm_heatmap = plot_confusion_matrix_heatmap(test_targets, test_preds, CONFIG["class_names"],
                                                   CONFIG["report_dir"], model_name)

        # 保存错误样本
        print("\n📊 保存错误样本...")
        save_error_samples(error_samples, CONFIG["report_dir"], model_name)

        # 记录测试结果
        overall_results[model_name] = {
            "acc": test_acc,
            "macro_precision": macro_precision,
            "macro_f1": macro_f1,
            "report_df": report_df,
            "cm": cm,
            "cm_heatmap": cm_heatmap,
            "error_samples": error_samples
        }

        print(f"\n{'=' * 80}")
        print(f"✅ {model_name.upper()} 测试完成")
        print(f"整体准确率: {test_acc:.4f}")
        print(f"平均准确率 (宏平均): {macro_precision:.4f}")
        print(f"宏平均F1: {macro_f1:.4f}")
        print(f"报告保存路径: {CONFIG['report_dir']}")
        print(f"{'=' * 80}")

    # 生成汇总报告
    if len(selected_models) > 1:
        print(f"\n{'=' * 83}")
        print(f"📊 汇总报告")
        print(f"{'=' * 83}")

        # 准备汇总数据
        summary_data = []
        for model_name in selected_models:
            if model_name in overall_results:
                result = overall_results[model_name]
                summary_data.append({
                    "模型": model_name.upper(),
                    "整体准确率": result["acc"],
                    "平均准确率": result["macro_precision"],
                    "宏平均F1": result["macro_f1"]
                })

        # 创建汇总DataFrame
        summary_df = pd.DataFrame(summary_data)

        # 保存汇总报告
        summary_path = os.path.join(CONFIG["report_dir"], "汇总报告.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

        # 打印汇总结果
        print("\n📊 各模型性能对比:")
        print("-" * 65)
        for _, row in summary_df.iterrows():
            print(f"{row['模型']}:")
            print(f"  整体准确率: {row['整体准确率']:.4f}")
            print(f"  平均准确率 (宏平均): {row['平均准确率']:.4f}")
            print(f"  宏平均F1: {row['宏平均F1']:.4f}")

        # 计算平均值
        avg_acc = summary_df["整体准确率"].mean()
        avg_precision = summary_df["平均准确率"].mean()
        avg_f1 = summary_df["宏平均F1"].mean()

        print("-" * 65)
        print(f"平均整体准确率: {avg_acc:.4f}")
        print(f"平均准确率 (宏平均): {avg_precision:.4f}")
        print(f"平均宏平均F1: {avg_f1:.4f}")
        print("=" * 65)

        print(f"✅ 汇总报告已保存: {summary_path}")

    print(f"\n🎉 所有模型测试完成！")
    print(f"⏰ 总耗时: {time.time() - start_time:.1f}秒")
    print(f"👋 测试结束")


if __name__ == "__main__":
    main()