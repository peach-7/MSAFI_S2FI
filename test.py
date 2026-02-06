# test.py
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
from tqdm import tqdm
from utils.data_processing import FruitVideoDataset
from models.C3D import C3D
from models.I3D import I3D
from models.R3D import R3D
from models.MC3D import MC3D
from models.Transformer import Transformer
from models.Mamba_official import Mamba_official
from models.transformer_based_on_i3d import TransformerBasedOnI3D
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


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
    elif model_name == "transformer_based_on_i3d":  # 修正了拼写错误
        return TransformerBasedOnI3D(num_classes=config["num_classes"], num_frames=config["num_frames"],
                              input_size=config["input_size"])
    else:
        raise ValueError(f"不支持的模型：{model_name}")



def load_best_model(model_name, config):
    model = get_model(model_name, config)
    model_path = os.path.join(os.path.dirname(config["model_paths"][model_name]), f"best_{model_name}_model.pth")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=config["device"], weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 成功加载最佳模型: {model_path}")
        return model
    else:
        print(f"❌ 错误：找不到最佳模型文件: {model_path}")
        sys.exit(1)


def evaluate_model(model, test_loader, config):
    model.eval()
    all_predictions = []
    all_targets = []
    class_names = config['class_names']

    with torch.no_grad():
        print("🔍 开始模型评估...")
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="评估中")):
            data = data.to(config["device"])
            targets = targets.to(config["device"])

            outputs = model(data)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 计算指标
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    f1_macro = f1_score(all_targets, all_predictions, average='macro')

    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)

    # 分类报告
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=class_names,
        output_dict=True
    )

    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'classification_report': report
    }


def save_results(results, model_name, config):
    # 创建报告目录
    os.makedirs(config["report_dir"], exist_ok=True)

    # 1. 保存详细分类报告为CSV（指定UTF-8编码）
    report_df = pd.DataFrame(results['classification_report']).transpose()
    report_csv_path = os.path.join(config["report_dir"], f"{model_name}_classification_report.csv")
    report_df.to_csv(report_csv_path, encoding='utf-8-sig', index=True)  # 使用utf-8-sig编码解决Excel乱码
    print(f"📊 详细分类报告已保存至: {report_csv_path}")

    # 2. 保存混淆矩阵为CSV（指定UTF-8编码）
    cm_df = pd.DataFrame(
        results['confusion_matrix'],
        index=config['class_names'],
        columns=config['class_names']
    )
    cm_csv_path = os.path.join(config["report_dir"], f"{model_name}_confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path, encoding='utf-8-sig')  # 使用utf-8-sig编码
    print(f"📊 混淆矩阵已保存至: {cm_csv_path}")

    # 3. 绘制并保存混淆矩阵热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=config['class_names'],
        yticklabels=config['class_names']
    )
    plt.title(f'{model_name.upper()} 模型混淆矩阵', fontsize=16)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_plot_path = os.path.join(config["report_dir"], f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ 混淆矩阵热力图已保存至: {cm_plot_path}")

    # 4. 绘制并保存分类报告柱状图
    metrics = ['precision', 'recall', 'f1-score']
    class_names = config['class_names']

    # 准备数据
    data_for_plot = {metric: [] for metric in metrics}
    for class_name in class_names:
        for metric in metrics:
            data_for_plot[metric].append(results['classification_report'][class_name][metric])

    # 添加总体平均值
    class_names.append('平均值')
    for metric in metrics:
        data_for_plot[metric].append(results['classification_report']['macro avg'][metric])

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width, data_for_plot['precision'], width, label='精确率', alpha=0.8)
    bars2 = ax.bar(x, data_for_plot['recall'], width, label='召回率', alpha=0.8)
    bars3 = ax.bar(x + width, data_for_plot['f1-score'], width, label='F1分数', alpha=0.8)

    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title(f'{model_name.upper()} 模型分类性能报告', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 在柱子上添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    plt.tight_layout()

    report_plot_path = os.path.join(config["report_dir"], f"{model_name}_performance_report.png")
    plt.savefig(report_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼️ 性能报告柱状图已保存至: {report_plot_path}")


def print_evaluation_results(results, model_name):
    print("\n" + "=" * 60)
    print(f"🎯 {model_name.upper()} 模型评估结果")
    print("=" * 60)
    print(f"整体准确率: {results['accuracy']:.4f}")
    print(f"F1 Macro平均分: {results['f1_macro']:.4f}")

    print("\n详细分类报告:")
    print("-" * 60)
    report_df = pd.DataFrame(results['classification_report']).transpose()
    # 只打印数值部分，避免重复打印类别名称
    for idx, row in report_df.iterrows():
        if idx in ['accuracy', 'macro avg', 'weighted avg']:
            print(
                f"{idx:<12} precision: {row['precision']:.4f} recall: {row['recall']:.4f} f1-score: {row['f1-score']:.4f} support: {int(row['support'])}")
        else:
            print(
                f"{idx:<12} precision: {row['precision']:.4f} recall: {row['recall']:.4f} f1-score: {row['f1-score']:.4f} support: {int(row['support'])}")

    print(f"\n混淆矩阵:")
    print("-" * 60)
    cm_df = pd.DataFrame(
        results['confusion_matrix'],
        index=[f'真实_{name}' for name in ['好果', '次果', '烂果']],
        columns=[f'预测_{name}' for name in ['好果', '次果', '烂果']]
    )
    print(cm_df)


def main():
    print("=" * 80)
    print("🍎 测试水果分类模型")
    print("=" * 80)

    # 加载配置
    from config import CONFIG
    config = CONFIG

    print(f"📁 测试数据集: {config['test_data_dir']}")
    print(f"🔧 设备: {config['device'].upper()}")
    print(f"📊 帧数: {config['num_frames']} | 图像大小: {config['input_size']}")

    # 选择模型
    print("\n" + "=" * 60)
    print("📌 模型选择界面")
    print("=" * 60)
    for idx, name in enumerate(config["all_model_names"]):
        print(f"{idx + 1}. {name.upper()}")
    print(f"{len(config['all_model_names']) + 1}. 退出")
    print("=" * 60)

    while True:
        try:
            choice = input("\n请选择模型序号: ").strip()
            if not choice.isdigit():
                print("⚠️ 请输入数字序号")
                continue
            choice = int(choice)
            if choice == len(config["all_model_names"]) + 1:
                print("👋 退出程序")
                sys.exit(0)
            elif 1 <= choice <= len(config["all_model_names"]):
                model_name = config["all_model_names"][choice - 1]
                print(f"\n✅ 已选择: {model_name.upper()}")
                break
            else:
                print(f"⚠️ 无效序号，请输入1-{len(config['all_model_names']) + 1}之间的数字")
        except KeyboardInterrupt:
            print("\n👋 程序中断")
            sys.exit(0)
        except ValueError:
            print("⚠️ 请输入有效数字")

    # 加载测试数据集
    print("\n[1/3] 加载测试数据集...")
    test_dataset = FruitVideoDataset(
        data_dir=config["test_data_dir"],
        model_name=model_name,
        num_frames=config["num_frames"],
        input_size=config["input_size"],
        train=False
    )

    if len(test_dataset) == 0:
        print("❌ 错误：测试数据集为空，请检查路径")
        sys.exit(1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # 加载最佳模型
    print("\n[2/3] 加载最佳模型...")
    model = load_best_model(model_name, config)
    model.to(config["device"])

    # 评估模型
    print("\n[3/3] 开始模型评估...")
    results = evaluate_model(model, test_loader, config)

    # 保存结果
    print("\n[4/4] 保存评估结果...")
    save_results(results, model_name, config)

    # 打印结果
    print_evaluation_results(results, model_name)

    print(f"\n{'=' * 80}")
    print("✅ 测试完成！")
    print(f"📊 报告已保存至: {config['report_dir']}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()