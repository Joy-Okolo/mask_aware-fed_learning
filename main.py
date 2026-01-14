"""
Main entry point for CLAMP federated learning experiments
Runs experiments across multiple strategies and datasets
PATCHED VERSION - Saves results before plotting to prevent data loss
"""

import torch
import numpy as np
import random
import time
import json
import os
import sys

from config import get_config
from models import get_model
from data import create_federated_dataset
from utils import (
    GlobalConvergenceTracker,
    ResearchGradeStragglerTracker,
    ResearchGradeEfficiencyTracker,
    prepare_results_for_json,
    calculate_sdir
)
from federated import federated_training
from plots import plot_all_visualizations, create_summary_table


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def setup_environment():
    """Setup directories and environment"""
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")


def run_single_strategy(dataset_name, strategy, config, clients, client_speeds,
                       test_loader, device):
    """
    Run federated learning for a single strategy

    Args:
        dataset_name: Dataset name ('mnist', 'fashion-mnist', 'cifar10')
        strategy: Strategy name ('fedavg', 'fedpmt', 'feddrop', 'clamp')
        config: Configuration dictionary
        clients: List of client data loaders
        client_speeds: List of client speed factors
        test_loader: Global test set loader
        device: torch device

    Returns:
        Tuple of (results, straggler_tracker, efficiency_tracker)
    """
    print(f"\n{'='*80}")
    print(f"Running {strategy.upper()} on {dataset_name.upper()}")
    print(f"{'='*80}")

    # Reset seed for fair comparison
    set_seed(config['seed'])

    # Initialize global model
    global_model = get_model(dataset_name, config, device)

    # Initialize trackers
    convergence_tracker = GlobalConvergenceTracker(
        dataset_name=dataset_name,
        patience=config['patience'],
        accuracy_plateau=config['accuracy_plateau'],
        min_rounds=config['min_rounds']
    )

    straggler_tracker = ResearchGradeStragglerTracker(
        num_clients=len(clients),
        client_speeds=client_speeds,
        straggler_threshold=config['straggler_threshold']
    )

    efficiency_tracker = ResearchGradeEfficiencyTracker(
        num_clients=len(clients)
    )

    # Run federated training
    start_time = time.time()

    results = federated_training(
        clients=clients,
        client_speeds=client_speeds,
        test_loader=test_loader,
        global_model=global_model,
        device=device,
        strategy=strategy,
        config=config,
        straggler_tracker=straggler_tracker,
        efficiency_tracker=efficiency_tracker,
        convergence_tracker=convergence_tracker
    )

    experiment_time = time.time() - start_time

    # Print summary
    if results['test_accuracies']:
        final_acc = results['test_accuracies'][-1]
        rounds_to_conv = results.get('rounds_to_convergence', config['max_rounds'])
        early_stopped = results.get('early_stopped', False)
        conv_status = "CONVERGED" if early_stopped else "NO CONVERGENCE"

        print(f"\n{strategy.upper()} RESULTS:")
        print(f"   Final Accuracy: {final_acc:.2f}%")
        print(f"   Convergence Status: {conv_status}")
        print(f"   Rounds: {rounds_to_conv}")
        print(f"   Experiment Time: {experiment_time/60:.1f} minutes")
        print(f"   FLOPs: {results.get('flops_until_convergence', 0)/1e9:.2f} GFLOP")
        print(f"   Communication: {results.get('communication_until_convergence', 0)/1e6:.2f} MB")

    return results, straggler_tracker, efficiency_tracker


def save_single_result(result, strategy, dataset_name, config, timestamp, aggregation_type):
    """
    Save individual strategy result immediately after completion

    Args:
        result: Results dictionary for single strategy
        strategy: Strategy name
        dataset_name: Dataset name
        config: Configuration dictionary
        timestamp: Timestamp for filename
        aggregation_type: Aggregation type used
    """
    print(f"\n💾 Saving {strategy.upper()} results to JSON...")

    # Prepare serializable result
    serializable_result = prepare_results_for_json(result)

    # Create filename
    filename = f'./results/{dataset_name}_{strategy}_{aggregation_type}_{timestamp}.json'

    # Create result data
    result_data = {
        'strategy': strategy,
        'dataset': dataset_name,
        'aggregation_type': aggregation_type,
        'results': serializable_result,
        'config': {
            'num_clients': config['num_clients'],
            'clients_per_round': config['clients_per_round'],
            'max_rounds': config['max_rounds'],
            'convergence_threshold': config['convergence_threshold'],
            'patience': config['patience'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
        },
        'metadata': {
            'timestamp': timestamp,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'framework_version': 'CLAMP v2.0 (Extended)',
        }
    }

    # Save to file
    try:
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"✅ Results saved to: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return None


def run_experiment(dataset_name, strategies=None, aggregation_type='min_depth'):
    """
    Run complete experiment for a dataset

    Args:
        dataset_name: Dataset name ('mnist', 'fashion-mnist', 'cifar10')
        strategies: List of strategies to run (default: all)
        aggregation_type: 'min_depth' or 'mask_aware'

    Returns:
        Dictionary with all results
    """
    if strategies is None:
        strategies = ['fedavg', 'fedpmt', 'feddrop', 'clamp']

    print(f"\n{'#'*80}")
    print(f"# CLAMP FEDERATED LEARNING EXPERIMENT")
    print(f"# Dataset: {dataset_name.upper()}")
    print(f"# Aggregation: {aggregation_type.upper()}")
    print(f"# Strategies: {', '.join([s.upper() for s in strategies])}")
    print(f"{'#'*80}\n")

    # Setup
    setup_environment()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get configuration
    config = get_config(dataset_name)
    config['aggregation_type'] = aggregation_type  # Override aggregation type

    print(f"Configuration:")
    print(f"   Clients: {config['num_clients']}")
    print(f"   Clients per round: {config['clients_per_round']}")
    print(f"   Max rounds: {config['max_rounds']}")
    print(f"   Convergence threshold: {config['convergence_threshold']}%")
    print(f"   Aggregation: {aggregation_type}")

    # Create federated dataset
    print(f"\nCreating federated {dataset_name} dataset...")
    clients, test_loader, client_speeds = create_federated_dataset(
        dataset_name=dataset_name,
        num_clients=config['num_clients'],
        batch_size=config['batch_size'],
        iid=True,
        val_split=0.2
    )

    # Run all strategies
    all_results = []
    all_straggler_trackers = []
    all_efficiency_trackers = []
    saved_files = []

    timestamp = int(time.time())  # Single timestamp for all strategies in this run

    for strategy in strategies:
        results, straggler_tracker, efficiency_tracker = run_single_strategy(
            dataset_name=dataset_name,
            strategy=strategy,
            config=config,
            clients=clients,
            client_speeds=client_speeds,
            test_loader=test_loader,
            device=device
        )

        all_results.append(results)
        all_straggler_trackers.append(straggler_tracker)
        all_efficiency_trackers.append(efficiency_tracker)

        # 🔥 SAVE IMMEDIATELY AFTER EACH STRATEGY
        saved_file = save_single_result(
            results, strategy, dataset_name, config, timestamp, aggregation_type
        )
        if saved_file:
            saved_files.append(saved_file)

    # Save combined results
    print(f"\n💾 Saving combined results...")
    save_combined_results(all_results, strategies, dataset_name, config, timestamp, aggregation_type)

    # Generate visualizations (this might fail, but data is already saved)
    print(f"\n📊 Generating visualizations...")
    try:
        plot_all_visualizations(all_results, strategies, dataset_name, timestamp)
        print(f"✅ All visualizations created successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Plotting failed with error: {e}")
        print(f"✅ But all results were already saved to ./results/")
        print(f"\nYou can generate plots later using:")
        print(f"   python3 plot_enhanced.py {' '.join(saved_files)}")

    # Print summary table
    try:
        summary = create_summary_table(all_results, strategies, dataset_name)
        print(summary)
    except Exception as e:
        print(f"⚠️  Warning: Summary table generation failed: {e}")

    return {
        'results': all_results,
        'straggler_trackers': all_straggler_trackers,
        'efficiency_trackers': all_efficiency_trackers,
        'strategies': strategies,
        'config': config,
        'saved_files': saved_files
    }


def save_combined_results(all_results, strategies, dataset_name, config, timestamp, aggregation_type):
    """
    Save combined experiment results to JSON file

    Args:
        all_results: List of results dictionaries
        strategies: List of strategy names
        dataset_name: Dataset name
        config: Configuration dictionary
        timestamp: Timestamp for filename
        aggregation_type: Aggregation type used
    """
    # Prepare serializable results
    serializable_results = []
    for result in all_results:
        serializable_result = prepare_results_for_json(result)
        serializable_results.append(serializable_result)

    # Create comprehensive results dictionary
    results_data = {
        'experiment_results': {
            strategy: result for strategy, result in zip(strategies, serializable_results)
        },
        'experiment_config': {
            'dataset': dataset_name,
            'aggregation_type': aggregation_type,
            'num_clients': config['num_clients'],
            'clients_per_round': config['clients_per_round'],
            'max_rounds': config['max_rounds'],
            'convergence_threshold': config['convergence_threshold'],
            'patience': config['patience'],
            'strategies': strategies,
        },
        'metadata': {
            'timestamp': timestamp,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'framework_version': 'CLAMP v2.0 (Extended)',
        }
    }

    # Save to file
    filename = f'./results/{dataset_name}_{aggregation_type}_combined_{timestamp}.json'
    try:
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"✅ Combined results saved to: {filename}")
    except Exception as e:
        print(f"❌ Error saving combined results: {e}")


def compare_aggregation_methods(dataset_name, strategies=None):
    """
    Compare min_depth vs mask_aware aggregation

    Args:
        dataset_name: Dataset name
        strategies: List of strategies (default: ['clamp'])
    """
    if strategies is None:
        strategies = ['clamp']  # Focus on CLAMP for comparison

    print(f"\n{'#'*80}")
    print(f"# AGGREGATION METHOD COMPARISON")
    print(f"# Dataset: {dataset_name.upper()}")
    print(f"# Comparing: MIN_DEPTH vs MASK_AWARE")
    print(f"{'#'*80}\n")

    # Run with min_depth
    print("\n" + "="*80)
    print("RUNNING WITH MIN_DEPTH AGGREGATION")
    print("="*80)
    results_min_depth = run_experiment(dataset_name, strategies, aggregation_type='min_depth')

    # Run with mask_aware
    print("\n" + "="*80)
    print("RUNNING WITH MASK_AWARE AGGREGATION")
    print("="*80)
    results_mask_aware = run_experiment(dataset_name, strategies, aggregation_type='mask_aware')

    # Print comparison
    print(f"\n{'='*80}")
    print(f"AGGREGATION COMPARISON - {dataset_name.upper()}")
    print(f"{'='*80}")

    for strategy in strategies:
        min_depth_results = results_min_depth['results'][strategies.index(strategy)]
        mask_aware_results = results_mask_aware['results'][strategies.index(strategy)]

        print(f"\n{strategy.upper()}:")
        print(f"{'Metric':<30} {'MIN_DEPTH':<15} {'MASK_AWARE':<15} {'Improvement':<15}")
        print(f"{'-'*75}")

        # Accuracy
        acc_min = min_depth_results['test_accuracies'][-1] if min_depth_results['test_accuracies'] else 0
        acc_mask = mask_aware_results['test_accuracies'][-1] if mask_aware_results['test_accuracies'] else 0
        acc_diff = acc_mask - acc_min
        print(f"{'Final Accuracy (%)':<30} {acc_min:<15.2f} {acc_mask:<15.2f} {acc_diff:+.2f}")

        # Rounds
        rounds_min = min_depth_results.get('rounds_to_convergence', 'N/A')
        rounds_mask = mask_aware_results.get('rounds_to_convergence', 'N/A')
        if isinstance(rounds_min, int) and isinstance(rounds_mask, int):
            rounds_diff = rounds_mask - rounds_min
            print(f"{'Rounds to Convergence':<30} {rounds_min:<15} {rounds_mask:<15} {rounds_diff:+d}")

        # FLOPs
        flops_min = min_depth_results.get('flops_until_convergence', 0) / 1e9
        flops_mask = mask_aware_results.get('flops_until_convergence', 0) / 1e9
        flops_diff = ((flops_mask - flops_min) / max(flops_min, 1)) * 100
        print(f"{'Total FLOPs (GFLOP)':<30} {flops_min:<15.2f} {flops_mask:<15.2f} {flops_diff:+.1f}%")

        # Communication
        comm_min = min_depth_results.get('communication_until_convergence', 0) / 1e6
        comm_mask = mask_aware_results.get('communication_until_convergence', 0) / 1e6
        comm_diff = ((comm_mask - comm_min) / max(comm_min, 1)) * 100
        print(f"{'Total Comm (MB)':<30} {comm_min:<15.2f} {comm_mask:<15.2f} {comm_diff:+.1f}%")

        # SDIR
        sdir_min = min_depth_results.get('straggler_statistics', {}).get('avg_slow_inclusion_rate', 0) * 100
        sdir_mask = mask_aware_results.get('straggler_statistics', {}).get('avg_slow_inclusion_rate', 0) * 100
        sdir_diff = sdir_mask - sdir_min
        print(f"{'SDIR (%)':<30} {sdir_min:<15.1f} {sdir_mask:<15.1f} {sdir_diff:+.1f}")

    print(f"{'='*80}\n")


def main():
    """Main entry point"""
    # Parse command line arguments (simple version)
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = 'mnist'  # Default

    # Validate dataset
    valid_datasets = ['mnist', 'fashion-mnist', 'cifar10']
    if dataset_name not in valid_datasets:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Valid datasets: {valid_datasets}")
        sys.exit(1)

    # Run experiment with all strategies
    strategies = ['fedavg', 'fedpmt', 'feddrop', 'clamp']

    # Option 1: Run with single aggregation type
    results = run_experiment(dataset_name, strategies, aggregation_type='min_depth')

    # Option 2: Compare aggregation methods (uncomment to use)
    # compare_aggregation_methods(dataset_name, strategies=['clamp'])

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved to: ./results/")
    print(f"Plots saved to: ./plots/ (if plotting succeeded)")
    print("="*80 + "\n")

    if results.get('saved_files'):
        print("Individual result files:")
        for f in results['saved_files']:
            print(f"  - {f}")
        print(f"\nTo generate enhanced plots:")
        print(f"  python3 plot_enhanced.py {' '.join(results['saved_files'])}")


if __name__ == '__main__':
    main()
