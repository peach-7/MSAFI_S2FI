# config.py
import os
import torch

CONFIG = {
    "train_data_dir": "D:/Graduation Project/12.21dataset/train_丰树",
    "val_data_dir": "D:/Graduation Project/12.21dataset/val_丰树",
    "test_data_dir": "D:/Graduation Project/12.21dataset/test_丰树",
    "report_dir": "reports",
    "model_paths": {
        "r3d": "trained_models/best_r3d_model.pth",
        "i3d": "trained_models/best_i3d_model.pth",
        "c3d": "trained_models/best_c3d_model.pth",
        "mc3d": "trained_models/best_mc3d_model.pth",
        "transformer": "trained_models/best_transformer_model.pth",
        "mamba_official": "trained_models/best_mamba_official_model.pth"
    },
    "all_model_names": [
        "r3d", "i3d", "c3d", "mc3d", "transformer", "mamba_official"
    ],
    "num_classes": 3,
    "num_frames": 16,
    "input_size": 112,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "initial_batch_size": 8,  # 初始批量大小
    "batch_size_increase_factor": 1.1,  # 批量大小增加因子
    "max_batch_size": 64,  # 最大批量大小
    "batch_size": 32,  # 验证和测试的批量大小
    "num_workers": 4,
    "epochs": 100,
    "learning_rate": 0.001,
    "patience": 20,  # Early Stopping 的耐心值
    "class_names": ["好果", "次果", "烂果"],
    "target_count": 3000  # 新增的目标数量
}
