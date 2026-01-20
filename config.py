"""
Configuration file for CLAMP federated learning experiments
UPDATED: Match published paper configuration (100 clients, all participate)
"""

# Shared configuration across all datasets
DEFAULT_CONFIG = {
    'num_clients': 100,              # ✅ Match published
    'clients_per_round': 100,         # ✅ All clients participate
    'batch_size': 64,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    'straggler_threshold': 1.5,      # gamma in Eq. 3
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
        # CLAMP parameters (match published)
        'initial_depth': 3,
        'max_layers': 5,
        'fixed_depth': 3,
        'T_low': 8.0,
        'T_high': 25.0,
        # Aggregation
        'aggregation_type': 'mask_aware',  # or 'mask_aware'
        'min_participation_ratio': 0.5,
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
                                
