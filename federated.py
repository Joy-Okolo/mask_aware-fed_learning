"""
Core federated learning components
- Client local training
- Server aggregation strategies (min_depth and mask_aware)
- Federated training loop with support for multiple strategies
- CLAMP depth adaptation logic

FIXED: mask_aware aggregation now correctly uses selected_client_ids
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import random
import numpy as np
from collections import defaultdict

from models import apply_layer_mask


# =================== CLIENT LOCAL TRAINING ===================

def train_client(model, train_loader, val_loader, layers_to_update, device,
                client_speed=1.0, max_epochs=15, lr=0.001, patience=5):
    """
    Perform local training on client

    Args:
        model: Global model to train
        train_loader: Client's training data
        val_loader: Client's validation data
        layers_to_update: Number of layers to train (from the end)
        device: torch device
        client_speed: Speed factor for this client (affects simulated time)
        max_epochs: Maximum local epochs
        lr: Learning rate
        patience: Early stopping patience

    Returns:
        Tuple of (model_state_dict, elapsed_time, final_train_loss,
                 final_val_accuracy, epochs_trained, train_losses,
                 val_losses, val_accuracies)
    """
    model = copy.deepcopy(model).to(device)
    apply_layer_mask(model, layers_to_update)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    start_time = time.time()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []
    val_accuracies = []

    model.train()
    for epoch in range(max_epochs):
        epoch_train_loss = 0
        num_batches = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            num_batches += 1

            # Simulate slower device
            if client_speed < 1.0:
                time.sleep(0.001 * (1.0 - client_speed))

        avg_train_loss = epoch_train_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_accuracy = 100 * correct / max(total, 1)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        model.train()

        if patience_counter >= patience:
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    elapsed_time = (time.time() - start_time) / client_speed
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_accuracy = val_accuracies[-1] if val_accuracies else 0
    epochs_trained = len(train_losses)

    return (model.state_dict(), elapsed_time, final_train_loss,
            final_val_accuracy, epochs_trained, train_losses,
            val_losses, val_accuracies)


# =================== SERVER AGGREGATION ===================

def aggregate_min_depth(client_updates, client_depths, client_sizes):
    """
    Original CLAMP aggregation: Aggregate only up to minimum common depth (Eq. 13)

    Args:
        client_updates: List of model state dicts from clients
        client_depths: Dict mapping client_id -> depth trained
        client_sizes: List of dataset sizes for weighted averaging

    Returns:
        Tuple of (aggregated_weights, min_depth)
    """
    # Find minimum depth across all participating clients
    min_depth = min(client_depths.values())

    # Weighted averaging (FedAvg) up to min_depth
    total_size = sum(client_sizes)
    avg_weights = copy.deepcopy(client_updates[0])

    for key in avg_weights:
        avg_weights[key] = torch.zeros_like(avg_weights[key])

    for i, weights in enumerate(client_updates):
        weight = client_sizes[i] / total_size
        for key in avg_weights:
            if 'num_batches_tracked' in key:
                avg_weights[key] = weights[key].clone()
            elif 'running_mean' in key or 'running_var' in key:
                avg_weights[key] += weights[key].float() / len(client_updates)
            else:
                avg_weights[key] += weights[key] * weight

    return avg_weights, min_depth


def aggregate_mask_aware(client_updates, client_depths, client_sizes,
                        selected_client_ids, min_participation_ratio=0.5):
    """
    NEW: Mask-aware layer-wise aggregation (FIXED VERSION)
    Aggregates each layer using only clients that trained that layer

    Args:
        client_updates: List of model state dicts from clients (in order of selection)
        client_depths: Dict mapping client_id -> depth trained for THIS ROUND
        client_sizes: List of dataset sizes for weighted averaging (in order of selection)
        selected_client_ids: List of actual client IDs that participated this round
        min_participation_ratio: Minimum fraction of clients needed to update a layer

    Returns:
        Tuple of (aggregated_weights, max_depth)
    """
    num_clients = len(client_updates)
    min_clients_threshold = max(1, int(num_clients * min_participation_ratio))

    # Initialize aggregated weights with first client's structure
    aggregated_weights = copy.deepcopy(client_updates[0])

    # Get all layer names
    layer_names = list(aggregated_weights.keys())

    # Determine which layers are trainable layers (weights/biases)
    trainable_layer_names = [name for name in layer_names
                            if 'weight' in name or 'bias' in name]

    # Count total trainable layers (only count weights, not biases)
    total_layers = len([name for name in trainable_layer_names if 'weight' in name])

    # For each layer, aggregate from clients that trained it
    layer_count = 0
    for layer_name in layer_names:
        # Skip BatchNorm tracking parameters
        if 'num_batches_tracked' in layer_name:
            continue

        # Handle BatchNorm running statistics (always aggregate from all clients)
        if 'running_mean' in layer_name or 'running_var' in layer_name:
            aggregated_weights[layer_name] = torch.zeros_like(aggregated_weights[layer_name]).float()
            for client_update in client_updates:
                aggregated_weights[layer_name] += client_update[layer_name].float() / len(client_updates)
            continue

        # Skip non-trainable parameters
        if 'weight' not in layer_name and 'bias' not in layer_name:
            continue

        # Determine layer index (counting from the end)
        if 'weight' in layer_name:
            # This is a new layer
            current_layer_idx = layer_count
            layer_count += 1
        else:
            # This is a bias corresponding to the previous weight layer
            current_layer_idx = layer_count - 1

        # ===== FIX: Use selected_client_ids to get correct depths =====
        # Find clients that trained this layer (from the end)
        # CLAMP trains the LAST N layers (from the end)
        # Layer indexing: 0=last layer (shallowest), total_layers-1=first layer (deepest)
        # Reverse the layer index to count from end
        layer_from_end = total_layers - 1 - current_layer_idx

        participating_clients = []
        participating_sizes = []

        for idx, client_id in enumerate(selected_client_ids):  # FIX: Use actual client IDs
            client_depth = client_depths[client_id]  # FIX: Now gets correct depth

            # Client trained the last 'client_depth' layers
            # If layer_from_end < client_depth, this client trained this layer
            # Example: layer_from_end=0 (last layer), client_depth=2 (trains last 2)
            #          0 < 2, so YES, client trained it
            # Example: layer_from_end=3 (4th from end), client_depth=2 (trains last 2)
            #          3 >= 2, so NO, client didn't train it
            if layer_from_end < client_depth:
                participating_clients.append(idx)  # Store list index for client_updates
                participating_sizes.append(client_sizes[idx])

        # Only aggregate if enough clients participated
        if len(participating_clients) >= min_clients_threshold:
            # Weighted average using only participating clients
            total_participating_size = sum(participating_sizes)
            aggregated_weights[layer_name] = torch.zeros_like(aggregated_weights[layer_name])

            for i, idx in enumerate(participating_clients):
                weight = participating_sizes[i] / total_participating_size
                aggregated_weights[layer_name] += client_updates[idx][layer_name] * weight
        # else: keep previous global weights (no update for this layer)

    return aggregated_weights, max(client_depths.values())  # Return max depth for logging


def evaluate_global_model(model, test_loader, device):
    """
    Evaluate global model on test set

    Args:
        model: Global model
        test_loader: Test dataset loader
        device: torch device

    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / max(total, 1)
    avg_loss = total_loss / max(len(test_loader), 1)
    return float(accuracy), float(avg_loss)


# =================== CLAMP DEPTH ADAPTATION ===================

def adjust_clamp_depth(elapsed_time, current_depth, client_id, T_low, T_high, max_layers):
    """
    CLAMP layer adjustment rule (Eq. 12)

    Args:
        elapsed_time: Observed runtime from last round
        current_depth: Current training depth for this client
        client_id: Client identifier
        T_low: Lower threshold for increasing depth
        T_high: Upper threshold for decreasing depth
        max_layers: Maximum number of layers in the model

    Returns:
        New depth for next round
    """
    # Client-specific thresholds (slight variation per client)
    client_T_low = T_low * (0.7 + (client_id % 10) * 0.1)
    client_T_high = T_high * (0.8 + (client_id % 10) * 0.2)

    if elapsed_time > client_T_high and current_depth > 1:
        return current_depth - 1  # Decrease depth (too slow)
    elif elapsed_time < client_T_low and current_depth < max_layers:
        return current_depth + 1  # Increase depth (fast enough)
    else:
        return current_depth  # Stay at current depth


class CLAMPStabilityTracker:
    """
    CLAMP stability tracking to prevent oscillations
    Requires multiple consecutive suggestions before changing depth
    """
    def __init__(self, num_clients):
        self.adaptation_history = {cid: [] for cid in range(num_clients)}
        self.consecutive_suggestions = {
            cid: {'increase': 0, 'decrease': 0, 'stay': 0}
            for cid in range(num_clients)
        }
        self.performance_history = {cid: [] for cid in range(num_clients)}

    def should_adapt(self, client_id, current_depth, suggested_depth,
                    client_performance=None, stability_threshold=3):
        """
        Enhanced adaptation decision with stability control

        Args:
            client_id: Client identifier
            current_depth: Current training depth
            suggested_depth: Suggested new depth from adjustment rule
            client_performance: Optional validation accuracy
            stability_threshold: Number of consecutive suggestions needed

        Returns:
            New depth (may be same as current if not stable)
        """
        if client_performance is not None:
            self.performance_history[client_id].append(client_performance)

        if suggested_depth == current_depth:
            # Reset counters if suggestion is to stay
            self.consecutive_suggestions[client_id] = {
                'increase': 0, 'decrease': 0, 'stay': 0
            }
            return current_depth

        if suggested_depth > current_depth:
            # Suggestion to increase
            self.consecutive_suggestions[client_id]['increase'] += 1
            self.consecutive_suggestions[client_id]['decrease'] = 0
            self.consecutive_suggestions[client_id]['stay'] = 0

            if self.consecutive_suggestions[client_id]['increase'] >= stability_threshold:
                self.consecutive_suggestions[client_id]['increase'] = 0
                return suggested_depth

        elif suggested_depth < current_depth:
            # Suggestion to decrease
            self.consecutive_suggestions[client_id]['decrease'] += 1
            self.consecutive_suggestions[client_id]['increase'] = 0
            self.consecutive_suggestions[client_id]['stay'] = 0

            if self.consecutive_suggestions[client_id]['decrease'] >= stability_threshold:
                self.consecutive_suggestions[client_id]['decrease'] = 0
                return suggested_depth

        return current_depth  # Not enough consecutive suggestions yet


# =================== FEDERATED TRAINING LOOP ===================

def federated_training(clients, client_speeds, test_loader, global_model, device,
                      strategy, config, straggler_tracker, efficiency_tracker,
                      convergence_tracker):
    """
    Main federated training loop (Algorithm 1)

    Args:
        clients: List of (train_loader, val_loader, dataset_size) tuples
        client_speeds: List of speed factors
        test_loader: Global test set
        global_model: Initial global model
        device: torch device
        strategy: One of 'fedavg', 'fedpmt', 'feddrop', 'clamp'
        config: Configuration dictionary
        straggler_tracker: Straggler tracking instance
        efficiency_tracker: Efficiency tracking instance
        convergence_tracker: Convergence tracking instance

    Returns:
        Dictionary with comprehensive results
    """
    max_rounds = config['max_rounds']
    max_local_epochs = config['max_local_epochs']
    lr = config['learning_rate']
    max_layers = config['max_layers']
    fixed_depth = config['fixed_depth']
    initial_depth = config['initial_depth']
    T_low = config['T_low']
    T_high = config['T_high']
    aggregation_type = config.get('aggregation_type', 'min_depth')
    min_participation_ratio = config.get('min_participation_ratio', 0.5)
    clients_per_round = config['clients_per_round']

    num_clients = len(clients)

    # Results tracking
    results = {
        'test_accuracies': [],
        'test_losses': [],
        'training_times': defaultdict(list),
        'client_depths': defaultdict(list),
        'client_epochs_trained': defaultdict(list),
        'client_val_accuracies': defaultdict(list),
        'round_times': [],
        'losses': defaultdict(list),
        'convergence_info': {},
        'stragglers_per_round': [],
        'client_straggler_history': defaultdict(list),
        'straggler_statistics': {},
        'participation_analysis': [],
        'flops_per_round': [],
        'communication_costs_per_round': [],
        'energy_consumption_per_round': [],
        'computational_efficiency': [],
        'communication_efficiency': [],
        'client_communication_breakdown': defaultdict(list),
        'rounds_to_convergence': None,
        'flops_until_convergence': 0,
        'communication_until_convergence': 0,
        'energy_until_convergence': 0,
        'accuracy_at_convergence': None,
        'early_stopped': False
    }

    # Initialize client depths
    client_depths = {cid: initial_depth for cid in range(num_clients)}
    clamp_tracker = CLAMPStabilityTracker(num_clients) if strategy == 'clamp' else None

    print(f"\nStarting Federated Learning")
    print(f"Strategy: {strategy.upper()} | Aggregation: {aggregation_type}")
    print(f"Clients: {num_clients} | Max Rounds: {max_rounds}")
    print(f"Min Participation Ratio: {min_participation_ratio}")
    print("-" * 80)

    for r in range(max_rounds):
        round_start_time = time.time()
        print(f"\n--- Round {r+1}/{max_rounds} ---")

        # Sample clients for this round
        selected_clients = random.sample(range(num_clients), clients_per_round)

        local_weights = []
        local_sizes = []
        round_training_times = []
        round_flops = []
        round_comm_costs = []
        round_energy = []
        round_client_depths = {}

        # Client local training
        for cid in selected_clients:
            train_loader, val_loader, dataset_size = clients[cid]

            # Determine layer depth based on strategy
            if strategy == "fedavg":
                layer_depth = max_layers
            elif strategy == "fedpmt":
                layer_depth = fixed_depth
            elif strategy == "feddrop":
                layer_depth = random.randint(1, max_layers)
            elif strategy == "clamp":
                layer_depth = client_depths[cid]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            results['client_depths'][cid].append(layer_depth)
            round_client_depths[cid] = layer_depth

            # Start monitoring
            start_metrics = efficiency_tracker.start_client_monitoring(cid)

            # Local training
            training_results = train_client(
                global_model, train_loader, val_loader, layer_depth, device,
                client_speed=client_speeds[cid], max_epochs=max_local_epochs,
                lr=lr, patience=5
            )

            (local_state_dict, elapsed_time, final_train_loss,
             final_val_accuracy, epochs_trained, train_losses,
             val_losses, val_accuracies) = training_results

            # End monitoring
            efficiency_metrics = efficiency_tracker.end_client_monitoring(
                cid, start_metrics, global_model, local_state_dict,
                layer_depth, strategy
            )

            # Record metrics
            results['training_times'][cid].append(elapsed_time)
            results['losses'][cid].append(final_train_loss)
            results['client_epochs_trained'][cid].append(epochs_trained)
            results['client_val_accuracies'][cid].append(final_val_accuracy)
            results['client_communication_breakdown'][cid].append(
                efficiency_metrics['communication_cost']
            )

            round_training_times.append(elapsed_time)
            round_flops.append(efficiency_metrics['flops'])
            round_comm_costs.append(efficiency_metrics['communication_cost']['total_bytes'])
            round_energy.append(efficiency_metrics['energy'])

            # CLAMP depth adaptation
            if strategy == "clamp":
                suggested_depth = adjust_clamp_depth(
                    elapsed_time, layer_depth, cid, T_low, T_high, max_layers
                )
                new_depth = clamp_tracker.should_adapt(
                    cid, layer_depth, suggested_depth,
                    client_performance=final_val_accuracy,
                    stability_threshold=3
                )
                client_depths[cid] = new_depth

            local_weights.append(local_state_dict)
            local_sizes.append(dataset_size)

        # Straggler identification
        round_stragglers, straggler_count, participation_analysis = \
            straggler_tracker.identify_round_stragglers_with_participation(
                round_training_times, r, selected_clients
            )

        # Record round-level metrics
        results['stragglers_per_round'].append(straggler_count)
        results['participation_analysis'].append(participation_analysis)
        results['flops_per_round'].append(int(np.sum(round_flops)))
        results['communication_costs_per_round'].append(int(np.sum(round_comm_costs)))
        results['energy_consumption_per_round'].append(float(np.sum(round_energy)))

        # Calculate efficiency metrics
        if results['test_accuracies']:
            prev_acc = results['test_accuracies'][-1]
        else:
            prev_acc = 0

        current_flops = np.sum(round_flops)
        current_comm = np.sum(round_comm_costs)

        comp_efficiency = prev_acc / max(current_flops / 1e9, 1e-10)
        comm_efficiency = prev_acc / max(current_comm / 1e6, 1e-10)

        results['computational_efficiency'].append(comp_efficiency)
        results['communication_efficiency'].append(comm_efficiency)

        # ===== FIX: Pass selected_client_ids to mask_aware =====
        # Server aggregation
        if aggregation_type == 'mask_aware':
            global_weights, effective_depth = aggregate_mask_aware(
                local_weights, round_client_depths, local_sizes,
                selected_clients,  # FIX: Pass actual client IDs!
                min_participation_ratio
            )
            print(f"   Mask-Aware Aggregation: Using {len(selected_clients)} clients")
        else:  # min_depth
            global_weights, effective_depth = aggregate_min_depth(
                local_weights, round_client_depths, local_sizes
            )
            print(f"   Min-Depth Aggregation: Depth={effective_depth}")

        global_model.load_state_dict(global_weights)

        # Global evaluation
        test_acc, test_loss = evaluate_global_model(global_model, test_loader, device)
        results['test_accuracies'].append(test_acc)
        results['test_losses'].append(test_loss)

        # Print round summary
        median_time = float(np.median(round_training_times))
        print(f"   Test Accuracy: {test_acc:.2f}%")
        print(f"   Stragglers: {straggler_count}/{clients_per_round}")
        print(f"   Median Time: {median_time:.1f}s")
        print(f"   Effective Depth: {effective_depth}")

        # Print depth distribution for CLAMP
        if strategy == 'clamp' and r % 20 == 0:
            depth_dist = [round_client_depths[cid] for cid in selected_clients]
            print(f"   Depth Distribution: min={min(depth_dist)}, max={max(depth_dist)}, avg={np.mean(depth_dist):.1f}")

        # Check for convergence
        converged = convergence_tracker.update_accuracy(test_acc, r + 1)
        if converged:
            print(f"\n*** CONVERGENCE DETECTED at Round {r+1} ***")
            print(f"*** Accuracy: {test_acc:.2f}% ***")

            results['rounds_to_convergence'] = r + 1
            results['flops_until_convergence'] = int(np.sum(results['flops_per_round']))
            results['communication_until_convergence'] = int(
                np.sum(results['communication_costs_per_round'])
            )
            results['energy_until_convergence'] = float(
                np.sum(results['energy_consumption_per_round'])
            )
            results['accuracy_at_convergence'] = test_acc
            results['early_stopped'] = True
            break

        round_time = time.time() - round_start_time
        results['round_times'].append(round_time)

    # Final statistics
    if not convergence_tracker.converged:
        results['rounds_to_convergence'] = max_rounds
        results['flops_until_convergence'] = int(np.sum(results['flops_per_round']))
        results['communication_until_convergence'] = int(
            np.sum(results['communication_costs_per_round'])
        )
        results['energy_until_convergence'] = float(
            np.sum(results['energy_consumption_per_round'])
        )
        results['accuracy_at_convergence'] = results['test_accuracies'][-1] if results['test_accuracies'] else 0.0

    results['convergence_info'] = convergence_tracker.get_convergence_info()
    results['straggler_statistics'] = straggler_tracker.get_comprehensive_straggler_statistics()

    return results
