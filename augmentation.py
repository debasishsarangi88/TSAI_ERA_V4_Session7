"""
Data augmentation techniques for CIFAR-10 training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import AUGMENTATION_CONFIG, ALBUMENTATIONS_CONFIG, DATASET_CONFIG


class MixUp:
    """MixUp data augmentation as described in Zhang et al., 2017"""
    
    def __init__(self, alpha=None):
        if alpha is None:
            alpha = AUGMENTATION_CONFIG['mixup_alpha']
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class CutMix:
    """CutMix data augmentation"""
    
    def __init__(self, alpha=None):
        if alpha is None:
            alpha = AUGMENTATION_CONFIG['cutmix_alpha']
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        W = x.size(2)
        H = x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam


class RICAP:
    """Random Image Cropping and Patching as described in Takahashi et al., 2018"""
    
    def __init__(self, beta=None):
        if beta is None:
            beta = AUGMENTATION_CONFIG['ricap_beta']
        self.beta = beta

    def __call__(self, x, y):
        I_x, I_y = x.size()[2:]
        w = int(np.round(I_x * np.random.beta(self.beta, self.beta)))
        h = int(np.round(I_y * np.random.beta(self.beta, self.beta)))

        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]

        cropped_images = {}
        c_ = {}
        W_ = {}

        for k in range(4):
            index = torch.randperm(x.size(0)).to(x.device)
            x_k = np.random.randint(0, I_x - w_[k] + 1)
            y_k = np.random.randint(0, I_y - h_[k] + 1)
            cropped_images[k] = x[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
            c_[k] = y[index]
            W_[k] = w_[k] * h_[k] / (I_x * I_y)

        # Patch cropped images
        patched_images = torch.cat(
            (torch.cat((cropped_images[0], cropped_images[1]), 2),
             torch.cat((cropped_images[2], cropped_images[3]), 2)),
            3
        )

        return patched_images, c_, W_


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def ricap_criterion(criterion, pred, c_, W_):
    """RICAP loss function - FIXED: average instead of sum"""
    return sum([W_[k] * criterion(pred, c_[k]) for k in range(4)]) / 4.0


def get_albumentations_transforms():
    """Get Albumentations transforms for training and testing"""
    
    # Training transforms
    train_transform = A.Compose([
        A.HorizontalFlip(p=ALBUMENTATIONS_CONFIG['horizontal_flip_prob']),
        A.ShiftScaleRotate(
            shift_limit=ALBUMENTATIONS_CONFIG['shift_limit'],
            scale_limit=ALBUMENTATIONS_CONFIG['scale_limit'],
            rotate_limit=ALBUMENTATIONS_CONFIG['rotate_limit'],
            p=ALBUMENTATIONS_CONFIG['shift_scale_rotate_prob']
        ),
        A.RandomBrightnessContrast(
            brightness_limit=ALBUMENTATIONS_CONFIG['brightness_limit'],
            contrast_limit=ALBUMENTATIONS_CONFIG['contrast_limit'],
            p=ALBUMENTATIONS_CONFIG['brightness_contrast_prob']
        ),
        A.HueSaturationValue(
            hue_shift_limit=ALBUMENTATIONS_CONFIG['hue_shift_limit'],
            sat_shift_limit=ALBUMENTATIONS_CONFIG['sat_shift_limit'],
            val_shift_limit=ALBUMENTATIONS_CONFIG['val_shift_limit'],
            p=ALBUMENTATIONS_CONFIG['hue_saturation_prob']
        ),
        A.CoarseDropout(
            max_holes=ALBUMENTATIONS_CONFIG['max_holes'],
            max_height=ALBUMENTATIONS_CONFIG['max_height'],
            max_width=ALBUMENTATIONS_CONFIG['max_width'],
            min_holes=1,
            min_height=4,
            min_width=4,
            p=ALBUMENTATIONS_CONFIG['coarse_dropout_prob']
        ),
        A.Normalize(mean=DATASET_CONFIG['mean'], std=DATASET_CONFIG['std']),
        ToTensorV2()
    ])

    # Test transforms
    test_transform = A.Compose([
        A.Normalize(mean=DATASET_CONFIG['mean'], std=DATASET_CONFIG['std']),
        ToTensorV2()
    ])

    return train_transform, test_transform


def get_augmentation_techniques():
    """Get initialized augmentation techniques"""
    return {
        'mixup': MixUp(),
        'cutmix': CutMix(),
        'ricap': RICAP()
    }


def apply_augmentation(data, target, augmentation_techniques, probabilities=None):
    """Apply random augmentation technique to batch"""
    if probabilities is None:
        probabilities = {
            'mixup': AUGMENTATION_CONFIG['mixup_prob'],
            'cutmix': AUGMENTATION_CONFIG['cutmix_prob'],
            'ricap': AUGMENTATION_CONFIG['ricap_prob']
        }
    
    rand_val = np.random.rand()
    cumulative_prob = 0.0
    
    # RICAP
    cumulative_prob += probabilities['ricap']
    if rand_val < cumulative_prob:
        data, c_, W_ = augmentation_techniques['ricap'](data, target)
        return data, target, 'ricap', {'c_': c_, 'W_': W_}
    
    # MixUp
    cumulative_prob += probabilities['mixup']
    if rand_val < cumulative_prob:
        data, target_a, target_b, lam = augmentation_techniques['mixup'](data, target)
        return data, target, 'mixup', {'target_a': target_a, 'target_b': target_b, 'lam': lam}
    
    # CutMix
    cumulative_prob += probabilities['cutmix']
    if rand_val < cumulative_prob:
        data, target_a, target_b, lam = augmentation_techniques['cutmix'](data, target)
        return data, target, 'cutmix', {'target_a': target_a, 'target_b': target_b, 'lam': lam}
    
    # No augmentation
    return data, target, 'none', {}
