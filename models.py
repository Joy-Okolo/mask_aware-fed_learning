"""
Neural network models for federated learning experiments
- MNISTNet: MLP for MNIST
- FashionMNISTNet: MLP for Fashion-MNIST
- CIFAR10Net: CNN for CIFAR-10
"""

import torch
import torch.nn as nn


class MNISTNet(nn.Module):
    """Research-grade MNIST neural network architecture"""

    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128, 64],
                 num_classes=10, dropout_rate=0.3):
        super(MNISTNet, self).__init__()
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FashionMNISTNet(nn.Module):
    """Research-grade Fashion-MNIST neural network"""

    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128, 64],
                 num_classes=10, dropout_rate=0.4):
        super(FashionMNISTNet, self).__init__()
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CIFAR10Net(nn.Module):
    """Research-grade CIFAR-10 CNN architecture"""

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CIFAR10Net, self).__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def apply_layer_mask(model, layers_to_update):
    """
    Apply mask to update only the last 'layers_to_update' layers

    Args:
        model: Neural network model (MNISTNet, FashionMNISTNet, or CIFAR10Net)
        layers_to_update: Number of layers to keep trainable (from the end)
    """
    # For MLP models (MNIST, Fashion-MNIST)
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        all_layers = [module for module in model.model.modules()
                      if isinstance(module, nn.Linear)]

        total_linear_layers = len(all_layers)
        layers_to_freeze = max(0, total_linear_layers - layers_to_update)

        for i, layer in enumerate(all_layers):
            requires_grad = (i >= layers_to_freeze)
            if hasattr(layer, 'weight'):
                layer.weight.requires_grad = requires_grad
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.requires_grad = requires_grad

    # For CNN model (CIFAR-10)
    elif hasattr(model, 'features') and hasattr(model, 'classifier'):
        all_layers = []

        # Add conv layers
        for module in model.features.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                all_layers.append(module)

        # Add classifier layers
        for module in model.classifier.modules():
            if isinstance(module, nn.Linear):
                all_layers.append(module)

        total_layers = len(all_layers)
        layers_to_freeze = max(0, total_layers - layers_to_update)

        for i, layer in enumerate(all_layers):
            requires_grad = (i >= layers_to_freeze)
            for param in layer.parameters():
                param.requires_grad = requires_grad


def get_model(dataset_name, config, device):
    """
    Factory function to create model based on dataset

    Args:
        dataset_name: One of 'mnist', 'fashion-mnist', 'cifar10'
        config: Configuration dictionary
        device: torch device (cpu or cuda)

    Returns:
        Initialized model on the specified device
    """
    if dataset_name == 'mnist':
        model = MNISTNet(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )
    elif dataset_name == 'fashion-mnist':
        model = FashionMNISTNet(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )
    elif dataset_name == 'cifar10':
        model = CIFAR10Net(
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return model.to(device)
