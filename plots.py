"""
Visualization functions for federated learning experiments
Generates publication-quality plots comparing different strategies
FIXED VERSION - Handles array length mismatches gracefully
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os


def ensure_plots_directory():
    """Create plots directory if it doesn't exist"""
    os.makedirs('./plots', exist_ok=True)


def plot_accuracy_evolution(all_results, strategies, dataset_name, timestamp):
    """
    Plot test accuracy evolution over rounds

    Args:
        all_results: List of results dictionaries (one per strategy)
        strategies: List of strategy names
        dataset_name: Dataset name for title
        timestamp: Timestamp for filename
    """
    ensure_plots_directory()

    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['test_accuracies']:
            rounds = range(1, len(results['test_accuracies']) + 1)
            plt.plot(rounds, results['test_accuracies'],
                    label=strategy.upper(), color=colors[i], linewidth=2)

            # Mark convergence point if early stopped
            if results.get('early_stopped', False):
                conv_round = results.get('rounds_to_convergence', len(results['test_accuracies']))
                if conv_round <= len(results['test_accuracies']):
                    conv_acc = results['test_accuracies'][conv_round - 1]
                    plt.scatter(conv_round, conv_acc, color=colors[i],
                              s=100, marker='*', edgecolor='black', linewidth=1, zorder=5)

    # Add convergence threshold line
    thresholds = {'mnist': 95.0, 'fashion-mnist': 85.0, 'cifar10': 75.0}
    threshold = thresholds.get(dataset_name, 85.0)
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Convergence Threshold ({threshold}%)')

    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'{dataset_name.upper()} - Accuracy Evolution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f'./plots/{dataset_name}_accuracy_evolution_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def plot_accuracy_vs_time(all_results, strategies, dataset_name, timestamp):
    """
    Plot test accuracy vs cumulative time
    FIXED: Handles array length mismatches

    Args:
        all_results: List of results dictionaries
        strategies: List of strategy names
        dataset_name: Dataset name for title
        timestamp: Timestamp for filename
    """
    ensure_plots_directory()

    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['test_accuracies'] and results['round_times']:
            # FIX: Handle length mismatch between accuracies and times
            accuracies = results['test_accuracies']
            times = results['round_times']

            # Ensure both arrays have the same length
            min_len = min(len(accuracies), len(times))
            accuracies = accuracies[:min_len]
            times = times[:min_len]

            cumulative_time = np.cumsum(times)
            plt.plot(cumulative_time, accuracies,
                    label=strategy.upper(), color=colors[i], linewidth=2)

    plt.xlabel('Cumulative Time (seconds)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'{dataset_name.upper()} - Accuracy vs Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f'./plots/{dataset_name}_accuracy_vs_time_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def plot_stragglers_per_round(all_results, strategies, dataset_name, timestamp):
    """
    Plot number of stragglers per round

    Args:
        all_results: List of results dictionaries
        strategies: List of strategy names
        dataset_name: Dataset name for title
        timestamp: Timestamp for filename
    """
    ensure_plots_directory()

    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        if results['stragglers_per_round']:
            rounds = range(1, len(results['stragglers_per_round']) + 1)
            plt.plot(rounds, results['stragglers_per_round'],
                    label=strategy.upper(), color=colors[i], linewidth=1.5, alpha=0.7)

    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Number of Stragglers', fontsize=12)
    plt.title(f'{dataset_name.upper()} - Stragglers per Round', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f'./plots/{dataset_name}_stragglers_per_round_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def plot_convergence_comparison(all_results, strategies, dataset_name, timestamp):
    """
    Create a comprehensive convergence comparison figure with multiple subplots

    Args:
        all_results: List of results dictionaries
        strategies: List of strategy names
        dataset_name: Dataset name for title
        timestamp: Timestamp for filename
    """
    ensure_plots_directory()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{dataset_name.upper()} - Convergence Analysis',
                 fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # 1. Rounds to Convergence (top-left)
    ax1 = axes[0, 0]
    conv_rounds = []
    strategy_names = []
    bar_colors = []

    for i, (strategy, results) in enumerate(zip(strategies, all_results)):
        rounds = results.get('rounds_to_convergence',
                            results.get('max_rounds', len(results.get('test_accuracies', []))))
        conv_rounds.append(rounds)
        strategy_names.append(strategy.upper())
        bar_colors.append(colors[i])

    bars = ax1.bar(strategy_names, conv_rounds, color=bar_colors, alpha=0.8)
    ax1.set_ylabel('Rounds to Convergence', fontsize=11)
    ax1.set_title('Rounds to Convergence', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=0)

    for bar, val, result in zip(bars, conv_rounds, all_results):
        label = f'{val}' if result.get('early_stopped', False) else f'{val}*'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(conv_rounds)*0.01,
                label, ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 2. Communication Efficiency (top-right)
    ax2 = axes[0, 1]
    comm_eff = []

    for strategy, results in zip(strategies, all_results):
        final_acc = results['test_accuracies'][-1] if results['test_accuracies'] else 0
        total_comm = results.get('communication_until_convergence', 1)
        eff = final_acc / max(total_comm / 1e6, 1e-10)
        comm_eff.append(eff)

    bars = ax2.bar(strategy_names, comm_eff, color=bar_colors, alpha=0.8)
    ax2.set_ylabel('Accuracy / MB', fontsize=11)
    ax2.set_title('Communication Efficiency', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=0)

    for bar, val in zip(bars, comm_eff):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comm_eff)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 3. Computational Efficiency (bottom-left)
    ax3 = axes[1, 0]
    comp_eff = []

    for strategy, results in zip(strategies, all_results):
        final_acc = results['test_accuracies'][-1] if results['test_accuracies'] else 0
        total_flops = results.get('flops_until_convergence', 1)
        eff = final_acc / max(total_flops / 1e9, 1e-10)
        comp_eff.append(eff)

    bars = ax3.bar(strategy_names, comp_eff, color=bar_colors, alpha=0.8)
    ax3.set_ylabel('Accuracy / GFLOP', fontsize=11)
    ax3.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=0)

    for bar, val in zip(bars, comp_eff):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comp_eff)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 4. SDIR Comparison (bottom-right)
    ax4 = axes[1, 1]
    sdir_values = []

    for strategy, results in zip(strategies, all_results):
        straggler_stats = results.get('straggler_statistics', {})
        sdir = straggler_stats.get('avg_slow_inclusion_rate', 0) * 100
        sdir_values.append(sdir)

    bars = ax4.bar(strategy_names, sdir_values, color=bar_colors, alpha=0.8)
    ax4.set_ylabel('SDIR (%)', fontsize=11)
    ax4.set_title('Slow Device Inclusion Rate', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.tick_params(axis='x', rotation=0)

    for bar, val in zip(bars, sdir_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()

    filename = f'./plots/{dataset_name}_convergence_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def plot_efficiency_breakdown(all_results, strategies, dataset_name, timestamp):
    """
    Plot detailed efficiency breakdown

    Args:
        all_results: List of results dictionaries
        strategies: List of strategy names
        dataset_name: Dataset name for title
        timestamp: Timestamp for filename
    """
    ensure_plots_directory()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name.upper()} - Efficiency Breakdown',
                 fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    strategy_names = [s.upper() for s in strategies]

    # 1. Total FLOPs (top-left)
    ax1 = axes[0, 0]
    total_flops = []
    for results in all_results:
        flops = results.get('flops_until_convergence', 0) / 1e9  # Convert to GFLOP
        total_flops.append(flops)

    bars = ax1.bar(strategy_names, total_flops, color=colors[:len(strategies)], alpha=0.8)
    ax1.set_ylabel('Total FLOPs (GFLOP)', fontsize=11)
    ax1.set_title('Total Computation', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=0)

    # 2. Total Communication (top-right)
    ax2 = axes[0, 1]
    total_comm = []
    for results in all_results:
        comm = results.get('communication_until_convergence', 0) / 1e6  # Convert to MB
        total_comm.append(comm)

    bars = ax2.bar(strategy_names, total_comm, color=colors[:len(strategies)], alpha=0.8)
    ax2.set_ylabel('Total Communication (MB)', fontsize=11)
    ax2.set_title('Total Communication', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=0)

    # 3. Total Energy (bottom-left)
    ax3 = axes[1, 0]
    total_energy = []
    for results in all_results:
        energy = results.get('energy_until_convergence', 0) / 1000  # Convert to kJ
        total_energy.append(energy)

    bars = ax3.bar(strategy_names, total_energy, color=colors[:len(strategies)], alpha=0.8)
    ax3.set_ylabel('Total Energy (kJ)', fontsize=11)
    ax3.set_title('Total Energy Consumption', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=0)

    # 4. Total Time (bottom-right)
    ax4 = axes[1, 1]
    total_time = []
    for results in all_results:
        rounds_to_conv = results.get('rounds_to_convergence', len(results.get('round_times', [])))
        time = sum(results.get('round_times', [])[:rounds_to_conv])
        total_time.append(time)

    bars = ax4.bar(strategy_names, total_time, color=colors[:len(strategies)], alpha=0.8)
    ax4.set_ylabel('Total Time (seconds)', fontsize=11)
    ax4.set_title('Total Training Time', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=0)

    plt.tight_layout()

    filename = f'./plots/{dataset_name}_efficiency_breakdown_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def plot_all_visualizations(all_results, strategies, dataset_name, timestamp):
    """
    Generate all visualization plots
    FIXED: Wraps all plots in try-except to ensure partial success

    Args:
        all_results: List of results dictionaries (one per strategy)
        strategies: List of strategy names
        dataset_name: Dataset name for titles
        timestamp: Timestamp for filenames
    """
    print(f"\nGenerating visualizations for {dataset_name}...")

    plots = [
        ('accuracy_evolution', plot_accuracy_evolution),
        ('accuracy_vs_time', plot_accuracy_vs_time),
        ('stragglers_per_round', plot_stragglers_per_round),
        ('convergence_comparison', plot_convergence_comparison),
        ('efficiency_breakdown', plot_efficiency_breakdown)
    ]

    success_count = 0
    for name, plot_func in plots:
        try:
            plot_func(all_results, strategies, dataset_name, timestamp)
            success_count += 1
        except Exception as e:
            print(f"Warning: Failed to create {name}: {e}")

    print(f"Successfully created {success_count}/{len(plots)} plots")
    print(f"All visualizations saved to ./plots/")


def create_summary_table(all_results, strategies, dataset_name):
    """
    Create a text summary table of results

    Args:
        all_results: List of results dictionaries
        strategies: List of strategy names
        dataset_name: Dataset name

    Returns:
        Formatted string with summary table
    """
    summary = []
    summary.append(f"\n{'='*80}")
    summary.append(f"{dataset_name.upper()} - RESULTS SUMMARY")
    summary.append(f"{'='*80}")
    summary.append(f"{'Metric':<30} {'FedAvg':<12} {'FedPMT':<12} {'FedDrop':<12} {'CLAMP':<12}")
    summary.append(f"{'-'*80}")

    # Final Accuracy
    accs = [f"{r['test_accuracies'][-1]:.2f}%" if r['test_accuracies'] else "N/A"
            for r in all_results]
    summary.append(f"{'Final Accuracy':<30} {accs[0]:<12} {accs[1]:<12} {accs[2]:<12} {accs[3]:<12}")

    # Rounds to Convergence
    rounds = [f"{r.get('rounds_to_convergence', 'N/A')}" for r in all_results]
    summary.append(f"{'Rounds to Convergence':<30} {rounds[0]:<12} {rounds[1]:<12} {rounds[2]:<12} {rounds[3]:<12}")

    # FLOPs (GFLOP)
    flops = [f"{r.get('flops_until_convergence', 0)/1e9:.2f}" for r in all_results]
    summary.append(f"{'Total FLOPs (GFLOP)':<30} {flops[0]:<12} {flops[1]:<12} {flops[2]:<12} {flops[3]:<12}")

    # Communication (MB)
    comm = [f"{r.get('communication_until_convergence', 0)/1e6:.2f}" for r in all_results]
    summary.append(f"{'Total Comm (MB)':<30} {comm[0]:<12} {comm[1]:<12} {comm[2]:<12} {comm[3]:<12}")

    # SDIR (%)
    sdirs = [f"{r.get('straggler_statistics', {}).get('avg_slow_inclusion_rate', 0)*100:.1f}%"
             for r in all_results]
    summary.append(f"{'SDIR (%)':<30} {sdirs[0]:<12} {sdirs[1]:<12} {sdirs[2]:<12} {sdirs[3]:<12}")

    summary.append(f"{'='*80}\n")

    return '\n'.join(summary)
