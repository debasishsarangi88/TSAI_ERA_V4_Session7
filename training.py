"""
Training functions and utilities for CIFAR-10 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import numpy as np
import math
import time
from tqdm import tqdm
from collections import OrderedDict
import copy

from config import TRAINING_CONFIG, EMA_CONFIG, TTA_CONFIG
from augmentation import mixup_criterion, ricap_criterion


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    
    def __init__(self, smoothing=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        if smoothing is None:
            smoothing = 0.1  # Default value
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class EMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model, decay=None):
        self.model = model
        if decay is None:
            decay = EMA_CONFIG['decay']
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train_epoch_advanced(model, device, train_loader, optimizer, criterion, epoch,
                        augmentation_techniques, probabilities=None):
    """Train one epoch with advanced augmentation techniques"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if probabilities is None:
        probabilities = {
            'mixup': 0.3,
            'cutmix': 0.3,
            'ricap': 0.2
        }

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # Apply augmentation
        from augmentation import apply_augmentation
        data, target, aug_type, aug_params = apply_augmentation(
            data, target, augmentation_techniques, probabilities
        )

        # Forward pass
        outputs = model(data)

        # Calculate loss based on augmentation type
        if aug_type == 'ricap':
            loss = ricap_criterion(criterion, outputs, aug_params['c_'], aug_params['W_'])
        elif aug_type in ['mixup', 'cutmix']:
            loss = mixup_criterion(
                criterion, outputs, 
                aug_params['target_a'], aug_params['target_b'], 
                aug_params['lam']
            )
        else:
            loss = criterion(outputs, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAINING_CONFIG['gradient_clip_norm'])

        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'Aug': aug_type.title()
        })

    return running_loss / len(train_loader), 100. * correct / total


def test_epoch(model, device, test_loader, criterion):
    """Test the model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return test_loss / len(test_loader), 100. * correct / total


def train_model_advanced(model, device, train_loader, test_loader, 
                        augmentation_techniques, epochs=None, lr=None):
    """Train model with advanced techniques"""
    
    if epochs is None:
        epochs = TRAINING_CONFIG['epochs']
    if lr is None:
        lr = TRAINING_CONFIG['learning_rate']

    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # Optimizer with weight decay
    optimizer = optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=TRAINING_CONFIG['momentum'], 
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = TRAINING_CONFIG['warmup_epochs']
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA for model weights
    ema = EMA(model, decay=EMA_CONFIG['decay'])

    # Training history
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0.0

    print(f"Starting training for {epochs} epochs...")
    print(f"Initial learning rate: {lr}")
    print(f"Using device: {device}")
    print("=" * 60)

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch_advanced(
            model, device, train_loader, optimizer, criterion, epoch,
            augmentation_techniques
        )

        # Update EMA
        ema.update()

        # Test
        test_loss, test_acc = test_epoch(model, device, test_loader, criterion)

        # Apply EMA for testing
        ema.apply_shadow()
        test_loss_ema, test_acc_ema = test_epoch(model, device, test_loader, criterion)
        ema.restore()

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save best model - Use regular test accuracy, not EMA
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model_200k.pth')

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Print progress
        print(f'Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | '
              f'Test Acc (EMA): {test_acc_ema:.2f}% | LR: {current_lr:.6f}')

        # Early stopping check
        patience = TRAINING_CONFIG['early_stopping_patience']
        threshold = TRAINING_CONFIG['early_stopping_threshold']
        if epoch > patience and test_acc < max(test_accs[-patience:]) - threshold:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("=" * 60)
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")

    return train_losses, train_accs, test_losses, test_accs, best_acc


def test_time_augmentation(model, data, device, num_augmentations=None):
    """Enhanced Test Time Augmentation"""
    if num_augmentations is None:
        num_augmentations = TTA_CONFIG['num_augmentations']
        
    model.eval()
    predictions = []

    with torch.no_grad():
        # Original image
        pred = F.softmax(model(data), dim=1)
        predictions.append(pred)

        for _ in range(num_augmentations - 1):
            # Random horizontal flip
            if np.random.rand() > 0.5:
                data_aug = torch.flip(data, dims=[3])
            else:
                data_aug = data.clone()

            # Random rotation
            angle = np.random.uniform(*TTA_CONFIG['rotation_range'])
            data_aug = TF.rotate(
                data_aug, angle, 
                interpolation=TF.InterpolationMode.BILINEAR
            )

            # Random brightness and contrast
            brightness = np.random.uniform(*TTA_CONFIG['brightness_range'])
            contrast = np.random.uniform(*TTA_CONFIG['contrast_range'])
            data_aug = TF.adjust_brightness(data_aug, brightness)
            data_aug = TF.adjust_contrast(data_aug, contrast)

            pred = F.softmax(model(data_aug), dim=1)
            predictions.append(pred)

    # Average predictions
    final_pred = torch.mean(torch.stack(predictions), dim=0)
    return final_pred


def evaluate_with_tta(model, test_loader, device, num_augmentations=None):
    """Evaluate model with Test Time Augmentation"""
    if num_augmentations is None:
        num_augmentations = TTA_CONFIG['num_augmentations']
        
    model.eval()
    correct = 0
    total = 0

    print(f"Evaluating with TTA (num_augmentations={num_augmentations})...")

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="TTA Evaluation"):
            data, target = data.to(device), target.to(device)

            # Get TTA prediction
            pred = test_time_augmentation(model, data, device, num_augmentations)
            _, predicted = torch.max(pred, 1)

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    print(f"TTA Accuracy: {accuracy:.2f}%")
    return accuracy
