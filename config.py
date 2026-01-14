"""
Configuration file for CLAMP federated learning experiments
Contains all hyperparameters organized by dataset
"""

# Shared configuration across all datasets
DEFAULT_CONFIG = {
    'num_clients': 50,
    'clients_per_round': 10,
    'batch_size': 64,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    'straggler_threshold': 1.5,  # gamma in Eq. 3
    'seed': 42,
}

# Dataset-specific configurations
CONFIGS = {
    'mnist': {
        **DEFAULT_CONFIG,
        'max_rounds': 300,
        'max_local_epochs': 15,
        'convergence_threshold': 95.0,
        'patience': 15,
        'min_rounds': 50,
        'accuracy_plateau': 0.1,

        # Model architecture
        'input_size': 784,
        'hidden_sizes': [512, 256, 128, 64],
        'num_classes': 10,
        'dropout_rate': 0.3,

        # CLAMP parameters
        'initial_depth': 3,
        'max_layers': 5,
        'fixed_depth': 3,  # For FedPMT
        'T_low': 8.0,
        'T_high': 25.0,

        # Aggregation
        'aggregation_type': 'mask_aware',  # or 'mask_aware'
        'min_participation_ratio': 0.5,  # For mask-aware aggregation
    },

    'fashion-mnist': {
        **DEFAULT_CONFIG,
        'max_rounds': 400,
        'max_local_epochs': 20,
        'convergence_threshold': 85.0,
        'patience': 20,
        'min_rounds': 75,
        'accuracy_plateau': 0.15,

        # Model architecture
        'input_size': 784,
        'hidden_sizes': [512, 256, 128, 64],
        'num_classes': 10,
        'dropout_rate': 0.4,

        # CLAMP parameters
        'initial_depth': 3,
        'max_layers': 5,
        'fixed_depth': 3,
        'T_low': 12.0,
        'T_high': 35.0,

        # Aggregation
        'aggregation_type': 'mask_aware',
        'min_participation_ratio': 0.5,
    },

    'cifar10': {
        **DEFAULT_CONFIG,
        'batch_size': 64,
        'max_rounds': 400,
        'max_local_epochs': 25,
        'convergence_threshold': 75.0,
        'patience': 25,
        'min_rounds': 100,
        'accuracy_plateau': 0.2,
        'weight_decay': 5e-4,

        # Model architecture (CNN)
        'num_classes': 10,
        'dropout_rate': 0.5,

        # CLAMP parameters
        'initial_depth': 5,
        'max_layers': 10,
        'fixed_depth': 5,
        'T_low': 20.0,
        'T_high': 60.0,

        # Aggregation
        'aggregation_type': 'mask_aware',
        'min_participation_ratio': 0.5,
    },
}

def get_config(dataset_name):
    """
    Get configuration for a specific dataset

    Args:
        dataset_name: One of 'mnist', 'fashion-mnist', 'cifar10'

    Returns:
        Configuration dictionary
    """
    if dataset_name not in CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(CONFIGS.keys())}")

    return CONFIGS[dataset_name]
