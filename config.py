# config.py
import os
import torch

CONFIG = {
    "train_data_dir": "D:/dataset/train",
    "val_data_dir": "D:/dataset/valid",
    "test_data_dir": "D:/dataset/test",
    "report_dir": "reports",
    "model_paths": {
        "r3d": "trained_models/best_r3d_model.pth",
        "i3d": "trained_models/best_i3d_model.pth",
        "c3d": "trained_models/best_c3d_model.pth",
        "mc3d": "trained_models/best_mc3d_model.pth",
        "transformer": "trained_models/best_transformer_model.pth",
        "DVFLNet": "trained_models/best_DVFLNet_model.pth",
        "msafi": "trained_models/best_msafi_model.pth",
        "mamba": "trained_models/best_mamba_model.pth",
        "moclip_lite": "trained_models/best_moclip_lite_model.pth",
        "movinets": "trained_models/best_movinets_model.pth",
    },
    "all_model_names": [
        "r3d", "i3d", "c3d", "mc3d", "transformer",
        "videomamba", "DVFLNet", "msafi","mamba","moclip_lite","movinets"
    ],
    "num_classes": 3,
    "num_frames": 16,
    "input_size": 112,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "use_amp": True,
    "amp_dtype": torch.float16,

    "initial_batch_size": 32,
    "batch_size_increase_factor": 1.1,
    "max_batch_size": 64,
    "batch_size": 32,
    "num_workers": 8,
    "epochs": 100,
    "learning_rate": 0.0005,
    "patience": 30,
    "class_names": ["good", "poor", "bad"],
    "target_count": None,
    "class_weights": [0.76, 0.61, 21.34],
    "use_weighted_sampling": True,
    "bad_class_augmentation_factor": 10,
    "gradient_accumulation_steps": 2,
}