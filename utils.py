"""
Utility functions and tracking classes for federated learning experiments
- Convergence tracking with early stopping
- Straggler identification and participation analysis
- Efficiency monitoring (FLOPs, communication, energy)
- JSON serialization helpers
- Metric calculations
"""

import time
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


# =================== GLOBAL EARLY STOPPING FRAMEWORK ===================

class GlobalConvergenceTracker:
    """Research-grade global convergence tracking with early stopping"""

    def __init__(self, dataset_name, patience=15, accuracy_plateau=0.1, min_rounds=50):
        self.dataset_name = dataset_name
        self.patience = patience
        self.accuracy_plateau = accuracy_plateau
        self.min_rounds = min_rounds

        # Dataset-specific convergence thresholds
        self.convergence_thresholds = {
            'mnist': 95.0,
            'fashion-mnist': 85.0,
            'cifar10': 75.0
        }

        self.min_accuracy = self.convergence_thresholds.get(dataset_name, 85.0)

        # Tracking variables
        self.no_improvement_count = 0
        self.best_accuracy = 0.0
        self.convergence_round = None
        self.converged = False
        self.accuracy_history = []

    def update_accuracy(self, current_accuracy, round_num):
        """
        Update accuracy and check for convergence

        Args:
            current_accuracy: Test accuracy for current round
            round_num: Current round number (1-indexed)

        Returns:
            bool: True if convergence detected, False otherwise
        """
        self.accuracy_history.append(current_accuracy)

        # Don't check convergence until minimum rounds
        if round_num < self.min_rounds:
            return False

        # Check for improvement
        if current_accuracy > self.best_accuracy + self.accuracy_plateau:
            self.best_accuracy = current_accuracy
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Check convergence criteria
        accuracy_threshold_met = current_accuracy >= self.min_accuracy
        plateau_reached = self.no_improvement_count >= self.patience

        if accuracy_threshold_met and plateau_reached and not self.converged:
            self.convergence_round = round_num
            self.converged = True
            return True

        return False

    def get_convergence_info(self):
        """Get comprehensive convergence information"""
        return {
            'converged': self.converged,
            'convergence_round': self.convergence_round,
            'best_accuracy': self.best_accuracy,
            'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0,
            'min_accuracy_threshold': self.min_accuracy,
            'rounds_to_convergence': self.convergence_round if self.converged else None,
            'accuracy_at_convergence': self.best_accuracy if self.converged else None
        }


# =================== STRAGGLER TRACKING ===================

class ResearchGradeStragglerTracker:
    """Research-grade straggler tracking with participation analysis"""

    def __init__(self, num_clients, client_speeds, straggler_threshold=1.5):
        self.num_clients = num_clients
        self.client_speeds = client_speeds
        self.straggler_threshold = straggler_threshold

        # Track stragglers by capability vs performance
        self.round_stragglers = []
        self.client_straggler_history = {cid: [] for cid in range(num_clients)}
        self.total_stragglers_per_round = []

        # Enhanced participation tracking
        self.participation_metrics = {
            'slow_capable_clients': [i for i, speed in enumerate(client_speeds) if speed < 0.8],
            'normal_capable_clients': [i for i, speed in enumerate(client_speeds) if 0.8 <= speed <= 1.2],
            'fast_capable_clients': [i for i, speed in enumerate(client_speeds) if speed > 1.2],
        }

        self.accommodation_history = []


    def identify_round_stragglers_with_participation(self, client_times, round_num, selected_clients):
        """
        Enhanced straggler identification with participation analysis

        Args:
            client_times: List of completion times for selected clients (in order)
            round_num: Current round number
            selected_clients: List of client IDs that participated this round

        Returns:
            Tuple of (round_stragglers, straggler_count, participation_analysis)
        """
        median_time = float(np.median(client_times))
        threshold = median_time * self.straggler_threshold

        round_stragglers = []
        # IMPORTANT: Loop through indices, not client IDs directly
        for idx, cid in enumerate(selected_clients):
            time_taken = client_times[idx]  # Use idx (0,1,2...) not cid (5,42,...)
            is_straggler = time_taken > threshold
            round_stragglers.append(is_straggler)
            self.client_straggler_history[cid].append(is_straggler)

        straggler_count = int(sum(round_stragglers))
        self.round_stragglers.append(round_stragglers)
        self.total_stragglers_per_round.append(straggler_count)

        participation_analysis = self.analyze_participation_capability(
            client_times, threshold, selected_clients
        )
        self.accommodation_history.append(participation_analysis)

        return round_stragglers, straggler_count, participation_analysis
    def analyze_participation_capability(self, client_times, threshold, selected_clients):
        """
        Analyze algorithm's ability to accommodate vs exclude stragglers

        Args:
            client_times: List of completion times (in order of selected_clients)
            threshold: Time threshold for straggler identification
            selected_clients: List of client IDs that participated
        """
        slow_clients = set(self.participation_metrics['slow_capable_clients'])
        normal_clients = set(self.participation_metrics['normal_capable_clients'])
        fast_clients = set(self.participation_metrics['fast_capable_clients'])

        accommodation_threshold = threshold * 1.2

        successful_slow = 0
        successful_normal = 0
        successful_fast = 0

        # IMPORTANT: Loop through indices and selected_clients together
        for idx, cid in enumerate(selected_clients):
            time_taken = client_times[idx]  # Use idx, not cid!

            if cid in slow_clients:
                if time_taken <= accommodation_threshold:
                    successful_slow += 1
            elif cid in normal_clients:
                if time_taken <= threshold:
                    successful_normal += 1
            elif cid in fast_clients:
                if time_taken <= threshold:
                    successful_fast += 1

        # Count how many of each type participated
        participating_slow = len([cid for cid in selected_clients if cid in slow_clients])
        participating_normal = len([cid for cid in selected_clients if cid in normal_clients])
        participating_fast = len([cid for cid in selected_clients if cid in fast_clients])

        total_slow = len(slow_clients)
        total_normal = len(normal_clients)
        total_fast = len(fast_clients)

        return {
            'total_clients': self.num_clients,
            'slow_capable_clients': total_slow,
            'normal_capable_clients': total_normal,
            'fast_capable_clients': total_fast,
            'successful_slow_participation': successful_slow,
            'successful_normal_participation': successful_normal,
            'successful_fast_participation': successful_fast,
            'slow_inclusion_rate': successful_slow / max(participating_slow, 1),
            'normal_inclusion_rate': successful_normal / max(participating_normal, 1),
            'fast_inclusion_rate': successful_fast / max(participating_fast, 1),
            'stragglers_accommodated': successful_slow,
            'stragglers_excluded': participating_slow - successful_slow,
            'accommodation_efficiency': successful_slow / max(participating_slow, 1)
        }


    def get_comprehensive_straggler_statistics(self):
        """Research-grade comprehensive straggler statistics"""
        if not self.total_stragglers_per_round:
            return {}

        avg_stragglers = float(np.mean(self.total_stragglers_per_round))
        straggler_rate = avg_stragglers / self.num_clients

        if self.accommodation_history:
            avg_accommodation = float(np.mean([h['accommodation_efficiency']
                                              for h in self.accommodation_history]))
            avg_slow_inclusion = float(np.mean([h['slow_inclusion_rate']
                                               for h in self.accommodation_history]))
        else:
            avg_accommodation = 0.0
            avg_slow_inclusion = 0.0

        persistent_stragglers = self.get_persistent_stragglers()

        return {
            'avg_stragglers_per_round': avg_stragglers,
            'max_stragglers_in_round': int(max(self.total_stragglers_per_round)),
            'min_stragglers_in_round': int(min(self.total_stragglers_per_round)),
            'straggler_rate': straggler_rate,
            'persistent_stragglers': persistent_stragglers,
            'straggler_variance': float(np.var(self.total_stragglers_per_round)),
            'avg_accommodation_efficiency': avg_accommodation,
            'avg_slow_inclusion_rate': avg_slow_inclusion,
            'total_slow_devices': len(self.participation_metrics['slow_capable_clients']),
            'accommodation_success_score': avg_accommodation * avg_slow_inclusion
        }

    def get_persistent_stragglers(self, persistence_threshold=0.7):
        """Identify clients that are frequently stragglers"""
        persistent = []
        for cid, history in self.client_straggler_history.items():
            if len(history) > 0:
                rate = sum(history) / len(history)
                if rate >= persistence_threshold:
                    persistent.append(int(cid))
        return persistent


# =================== EFFICIENCY TRACKING ===================

class ResearchGradeEfficiencyTracker:
    """Research-grade efficiency tracking for federated learning"""

    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.client_flops = defaultdict(list)
        self.client_memory_usage = defaultdict(list)
        self.client_energy_estimate = defaultdict(list)
        self.client_communication_costs = defaultdict(list)
        self.total_communication_per_round = []
        self.communication_breakdown = defaultdict(list)
        self.convergence_efficiency = {}
        self.parameter_efficiency = defaultdict(list)

    def start_client_monitoring(self, client_id):
        """Start monitoring computational resources"""
        return {
            'start_time': time.time(),
            'start_memory': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }

    def calculate_mlp_training_flops(self, model, batch_size, active_layers_count):
        """Calculate MLP training FLOPs following ML literature standards"""
        total_flops = 0
        linear_layers = [module for module in model.model.modules()
                        if isinstance(module, nn.Linear)]

        active_layers = (linear_layers[:active_layers_count]
                        if active_layers_count <= len(linear_layers)
                        else linear_layers)

        for layer in active_layers:
            forward_flops = batch_size * layer.in_features * layer.out_features
            backward_flops = 2 * forward_flops
            layer_flops = forward_flops + backward_flops
            total_flops += layer_flops

        return int(total_flops)

    def calculate_cnn_training_flops(self, model, batch_size, active_layers_count,
                                    input_size=(3, 32, 32)):
        """Calculate CNN training FLOPs for CIFAR-10"""
        total_flops = 0
        current_size = input_size
        layer_count = 0

        for module in model.features.modules():
            if layer_count >= active_layers_count:
                break

            if isinstance(module, nn.Conv2d):
                # Conv2d FLOPs calculation
                out_h = ((current_size[1] + 2 * module.padding[0] - module.kernel_size[0])
                        // module.stride[0] + 1)
                out_w = ((current_size[2] + 2 * module.padding[1] - module.kernel_size[1])
                        // module.stride[1] + 1)

                kernel_flops = (module.kernel_size[0] * module.kernel_size[1] *
                               module.in_channels)
                output_elements = batch_size * module.out_channels * out_h * out_w

                forward_flops = kernel_flops * output_elements
                backward_flops = 2 * forward_flops
                total_flops += forward_flops + backward_flops

                current_size = (module.out_channels, out_h, out_w)
                layer_count += 1

            elif isinstance(module, nn.MaxPool2d):
                out_h = (current_size[1] - module.kernel_size) // module.stride + 1
                out_w = (current_size[2] - module.kernel_size) // module.stride + 1
                current_size = (current_size[0], out_h, out_w)

        # Add classifier FLOPs
        if hasattr(model, 'classifier') and active_layers_count > layer_count:
            for module in model.classifier.modules():
                if isinstance(module, nn.Linear):
                    forward_flops = batch_size * module.in_features * module.out_features
                    backward_flops = 2 * forward_flops
                    total_flops += forward_flops + backward_flops

        return int(total_flops)

    def calculate_federated_communication_cost(self, strategy, full_model_state_dict,
                                               active_layers_count):
        """Calculate bidirectional communication cost with protocol overhead"""
        full_model_size = 0
        for param in full_model_state_dict.values():
            full_model_size += param.numel() * 4

        # Calculate active model size based on layers
        active_model_size = 0
        linear_layers = [name for name in full_model_state_dict.keys()
                        if 'weight' in name or 'bias' in name]

        total_layers = len([name for name in linear_layers if 'weight' in name])

        if total_layers > 0:
            layers_to_include = min(active_layers_count, total_layers)
            layers_to_skip = max(0, total_layers - layers_to_include)

            layer_count = 0
            for name, param in full_model_state_dict.items():
                if 'weight' in name:
                    if layer_count >= layers_to_skip:
                        active_model_size += param.numel() * 4
                        bias_name = name.replace('weight', 'bias')
                        if bias_name in full_model_state_dict:
                            active_model_size += full_model_state_dict[bias_name].numel() * 4
                    layer_count += 1

        # Strategy-specific communication
        if strategy == "fedavg":
            downstream_bytes = full_model_size
            upstream_bytes = full_model_size
        elif strategy in ["fedpmt", "feddrop", "clamp"]:
            downstream_bytes = full_model_size
            upstream_bytes = active_model_size
        else:
            downstream_bytes = full_model_size
            upstream_bytes = full_model_size

        protocol_overhead = (downstream_bytes + upstream_bytes) * 0.05
        total_communication = downstream_bytes + upstream_bytes + protocol_overhead

        return {
            'downstream_bytes': int(downstream_bytes),
            'upstream_bytes': int(upstream_bytes),
            'protocol_overhead': int(protocol_overhead),
            'total_bytes': int(total_communication),
            'full_model_size': int(full_model_size),
            'active_model_size': int(active_model_size)
        }

    def end_client_monitoring(self, client_id, start_metrics, model, model_state_dict,
                             layer_depth, strategy, batch_size=64):
        """End monitoring and calculate comprehensive efficiency metrics"""
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        training_time = end_time - start_metrics['start_time']
        memory_used = abs(end_memory - start_metrics['start_memory'])

        # Calculate FLOPs based on model type
        if hasattr(model, 'model'):  # MLP (MNIST, Fashion-MNIST)
            flops = self.calculate_mlp_training_flops(model, batch_size, layer_depth)
            energy_multiplier = 40.0
        else:  # CNN (CIFAR-10)
            flops = self.calculate_cnn_training_flops(model, batch_size, layer_depth)
            energy_multiplier = 60.0

        comm_cost = self.calculate_federated_communication_cost(
            strategy, model_state_dict, layer_depth
        )

        estimated_energy = training_time * energy_multiplier

        # Record metrics
        self.client_flops[client_id].append(flops)
        self.client_memory_usage[client_id].append(int(memory_used))
        self.client_energy_estimate[client_id].append(estimated_energy)
        self.client_communication_costs[client_id].append(comm_cost['total_bytes'])

        return {
            'training_time': training_time,
            'flops': flops,
            'communication_cost': comm_cost,
            'energy': estimated_energy,
            'memory_used': int(memory_used)
        }


# =================== JSON SERIALIZATION ===================

def safe_serialize_value(value):
    """Recursively convert values to JSON-safe types"""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [safe_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): safe_serialize_value(v) for k, v in value.items()}
    elif isinstance(value, defaultdict):
        return {str(k): safe_serialize_value(v) for k, v in dict(value).items()}
    elif isinstance(value, set):
        return list(value)
    return value


def prepare_results_for_json(results):
    """Convert all results to JSON-serializable format"""
    return {str(key): safe_serialize_value(value) for key, value in results.items()}


# =================== METRIC CALCULATIONS ===================

def calculate_sdir(participation_history, slow_clients, total_rounds):
    """
    Calculate Slow-Device Inclusion Ratio (SDIR)

    Args:
        participation_history: Dict mapping round -> list of participating client IDs
        slow_clients: Set of slow client IDs
        total_rounds: Total number of rounds

    Returns:
        SDIR value between 0 and 1
    """
    if len(slow_clients) == 0:
        return 0.0

    inclusion_rates = []
    for round_num, participants in participation_history.items():
        included_slow = len(set(participants) & set(slow_clients))
        inclusion_rates.append(included_slow / len(slow_clients))

    return float(np.mean(inclusion_rates)) if inclusion_rates else 0.0


def calculate_efficiency_ratios(final_accuracy, total_flops, total_comm,
                                total_energy, total_time):
    """
    Calculate various efficiency ratios

    Returns:
        Dictionary with efficiency metrics
    """
    return {
        'computational_efficiency': final_accuracy / max(total_flops / 1e9, 1e-10),
        'communication_efficiency': final_accuracy / max(total_comm / 1e6, 1e-10),
        'energy_efficiency': final_accuracy / max(total_energy, 1e-10),
        'time_efficiency': final_accuracy / max(total_time, 1e-10)
    }
