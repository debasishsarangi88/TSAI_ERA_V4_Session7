"""
Configuration parameters for the CIFAR-10 model training.
"""

# Model Configuration
MODEL_CONFIG = {
    'num_classes': 10,
    'input_channels': 3,
    'initial_channels': 20,
    'channel_progression': [20, 40, 80, 160],
    'blocks_per_layer': 2,
    'dropout_rate': 0.2,
    'target_parameters': 200000,
    'target_receptive_field': 44,
    'target_accuracy': 85.0
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 200,
    'batch_size': 128,
    'learning_rate': 0.05,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'warmup_epochs': 5,
    'early_stopping_patience': 20,
    'early_stopping_threshold': 2.0,
    'gradient_clip_norm': 1.0
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'mixup_prob': 0.3,
    'cutmix_prob': 0.3,
    'ricap_prob': 0.2,
    'mixup_alpha': 1.0,
    'cutmix_alpha': 1.0,
    'ricap_beta': 0.3,
    'label_smoothing': 0.1
}

# Albumentations Configuration
ALBUMENTATIONS_CONFIG = {
    'horizontal_flip_prob': 0.5,
    'shift_scale_rotate_prob': 0.5,
    'brightness_contrast_prob': 0.5,
    'hue_saturation_prob': 0.5,
    'coarse_dropout_prob': 0.3,
    'shift_limit': 0.1,
    'scale_limit': 0.1,
    'rotate_limit': 15,
    'brightness_limit': 0.2,
    'contrast_limit': 0.2,
    'hue_shift_limit': 20,
    'sat_shift_limit': 30,
    'val_shift_limit': 20,
    'max_holes': 1,
    'max_height': 8,
    'max_width': 8
}

# CIFAR-10 Dataset Configuration
DATASET_CONFIG = {
    'name': 'CIFAR-10',
    'data_dir': './data',
    'num_workers': 2,
    'pin_memory': True,
    'mean': (0.4914, 0.4822, 0.4465),
    'std': (0.2023, 0.1994, 0.2010),
    'class_names': [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
}

# EMA Configuration
EMA_CONFIG = {
    'decay': 0.999,
    'use_for_model_selection': False  # Fixed: Use regular test accuracy
}

# TTA Configuration
TTA_CONFIG = {
    'num_augmentations': 8,
    'rotation_range': (-15, 15),
    'brightness_range': (0.8, 1.2),
    'contrast_range': (0.8, 1.2)
}

# Device Configuration
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_deterministic': True,
    'cuda_benchmark': False
}

# Random Seed Configuration
SEED_CONFIG = {
    'seed': 42,
    'set_all_seeds': True
}
