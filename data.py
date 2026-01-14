"""
Dataset loading and federated data partitioning
Handles MNIST, Fashion-MNIST, and CIFAR-10 datasets
Creates federated splits with heterogeneous client speeds
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms


def create_federated_mnist(num_clients=50, batch_size=64, iid=True, val_split=0.2):
    """
    Create federated MNIST dataset

    Args:
        num_clients: Number of federated clients
        batch_size: Batch size for training
        iid: If True, data is IID across clients; if False, non-IID
        val_split: Fraction of client data to use for validation

    Returns:
        Tuple of (clients, test_loader, client_speeds)
        - clients: List of (train_loader, val_loader, dataset_size) tuples
        - test_loader: Global test set loader
        - client_speeds: List of speed factors for each client
    """
    print("Downloading MNIST dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    total_samples = len(train_dataset)
    base_samples_per_client = total_samples // num_clients

    clients = []
    client_speeds = []

    print(f"Creating {num_clients} federated MNIST clients...")

    for client_id in range(num_clients):
        # Assign speed factors
        if client_id < num_clients * 0.1:
            speed_factor = np.random.uniform(0.3, 0.6)  # Very slow devices
        elif client_id < num_clients * 0.2:
            speed_factor = np.random.uniform(1.8, 2.5)  # Very fast devices
        else:
            speed_factor = np.random.uniform(0.8, 1.4)  # Normal devices

        client_samples = base_samples_per_client

        if iid:
            # IID: Random sampling
            indices = np.random.choice(range(total_samples), client_samples, replace=False)
        else:
            # Non-IID: Each client specializes in 2-4 digit classes
            num_categories = np.random.randint(2, 5)
            target_classes = np.random.choice(range(10), size=num_categories, replace=False)
            indices = []
            targets = np.array(train_dataset.targets)

            for target_class in target_classes:
                class_indices = np.where(targets == target_class)[0]
                class_samples = min(len(class_indices), client_samples // len(target_classes))
                selected = np.random.choice(class_indices, class_samples, replace=False)
                indices.extend(selected)

            indices = indices[:client_samples]

            if client_id < 5:
                client_specialization = [mnist_classes[i] for i in target_classes]
                print(f"Client {client_id} specializes in digits: {client_specialization}")

        # Create client dataset with train/val split
        client_dataset = Subset(train_dataset, indices)
        train_size = int((1 - val_split) * len(client_dataset))
        val_size = len(client_dataset) - train_size

        train_subset, val_subset = random_split(client_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        clients.append((train_loader, val_loader, len(client_dataset)))
        client_speeds.append(speed_factor)

    print(f"Created {num_clients} MNIST clients with {total_samples} total samples")
    return clients, test_loader, client_speeds


def create_federated_fashion_mnist(num_clients=50, batch_size=64, iid=True, val_split=0.2):
    """
    Create federated Fashion-MNIST dataset

    Args:
        num_clients: Number of federated clients
        batch_size: Batch size for training
        iid: If True, data is IID across clients; if False, non-IID
        val_split: Fraction of client data to use for validation

    Returns:
        Tuple of (clients, test_loader, client_speeds)
    """
    print("Downloading Fashion-MNIST dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    total_samples = len(train_dataset)
    base_samples_per_client = total_samples // num_clients

    clients = []
    client_speeds = []

    print(f"Creating {num_clients} federated Fashion-MNIST clients...")

    for client_id in range(num_clients):
        # Assign speed factors
        if client_id < num_clients * 0.1:
            speed_factor = np.random.uniform(0.25, 0.55)
        elif client_id < num_clients * 0.2:
            speed_factor = np.random.uniform(1.6, 2.3)
        else:
            speed_factor = np.random.uniform(0.75, 1.3)

        client_samples = base_samples_per_client

        if iid:
            indices = np.random.choice(range(total_samples), client_samples, replace=False)
        else:
            num_categories = np.random.randint(2, 5)
            target_classes = np.random.choice(range(10), size=num_categories, replace=False)
            indices = []
            targets = np.array(train_dataset.targets)

            for target_class in target_classes:
                class_indices = np.where(targets == target_class)[0]
                class_samples = min(len(class_indices), client_samples // len(target_classes))
                selected = np.random.choice(class_indices, class_samples, replace=False)
                indices.extend(selected)

            indices = indices[:client_samples]

            if client_id < 5:
                client_specialization = [fashion_classes[i] for i in target_classes]
                print(f"Client {client_id} specializes in: {client_specialization}")

        # Create client dataset with train/val split
        client_dataset = Subset(train_dataset, indices)
        train_size = int((1 - val_split) * len(client_dataset))
        val_size = len(client_dataset) - train_size

        train_subset, val_subset = random_split(client_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        clients.append((train_loader, val_loader, len(client_dataset)))
        client_speeds.append(speed_factor)

    print(f"Created {num_clients} Fashion-MNIST clients with {total_samples} total samples")
    return clients, test_loader, client_speeds


def create_federated_cifar10(num_clients=50, batch_size=64, iid=True, val_split=0.2):
    """
    Create federated CIFAR-10 dataset

    Args:
        num_clients: Number of federated clients
        batch_size: Batch size for training
        iid: If True, data is IID across clients; if False, non-IID
        val_split: Fraction of client data to use for validation

    Returns:
        Tuple of (clients, test_loader, client_speeds)
    """
    print("Downloading CIFAR-10 dataset...")

    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # No augmentation for test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']

    total_samples = len(train_dataset)
    base_samples_per_client = total_samples // num_clients

    clients = []
    client_speeds = []

    print(f"Creating {num_clients} federated CIFAR-10 clients...")

    for client_id in range(num_clients):
        # Assign speed factors (slower for CIFAR-10 due to CNN complexity)
        if client_id < num_clients * 0.1:
            speed_factor = np.random.uniform(0.2, 0.5)
        elif client_id < num_clients * 0.2:
            speed_factor = np.random.uniform(1.5, 2.0)
        else:
            speed_factor = np.random.uniform(0.7, 1.2)

        client_samples = base_samples_per_client

        if iid:
            indices = np.random.choice(range(total_samples), client_samples, replace=False)
        else:
            num_categories = np.random.randint(2, 5)
            target_classes = np.random.choice(range(10), size=num_categories, replace=False)
            indices = []
            targets = np.array(train_dataset.targets)

            for target_class in target_classes:
                class_indices = np.where(targets == target_class)[0]
                class_samples = min(len(class_indices), client_samples // len(target_classes))
                selected = np.random.choice(class_indices, class_samples, replace=False)
                indices.extend(selected)

            indices = indices[:client_samples]

            if client_id < 5:
                client_specialization = [cifar10_classes[i] for i in target_classes]
                print(f"Client {client_id} specializes in: {client_specialization}")

        # Create client dataset with train/val split
        client_dataset = Subset(train_dataset, indices)
        train_size = int((1 - val_split) * len(client_dataset))
        val_size = len(client_dataset) - train_size

        train_subset, val_subset = random_split(client_dataset, [train_size, val_size])

        # Use test transform for validation to avoid data leakage
        val_subset.dataset.transform = transform_test

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        clients.append((train_loader, val_loader, len(client_dataset)))
        client_speeds.append(speed_factor)

    print(f"Created {num_clients} CIFAR-10 clients with {total_samples} total samples")
    return clients, test_loader, client_speeds


def create_federated_dataset(dataset_name, num_clients=50, batch_size=64,
                             iid=True, val_split=0.2):
    """
    Unified function to create federated dataset for any supported dataset

    Args:
        dataset_name: One of 'mnist', 'fashion-mnist', 'cifar10'
        num_clients: Number of federated clients
        batch_size: Batch size for training
        iid: If True, data is IID across clients; if False, non-IID
        val_split: Fraction of client data to use for validation

    Returns:
        Tuple of (clients, test_loader, client_speeds)
    """
    if dataset_name == 'mnist':
        return create_federated_mnist(num_clients, batch_size, iid, val_split)
    elif dataset_name == 'fashion-mnist':
        return create_federated_fashion_mnist(num_clients, batch_size, iid, val_split)
    elif dataset_name == 'cifar10':
        return create_federated_cifar10(num_clients, batch_size, iid, val_split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Must be one of ['mnist', 'fashion-mnist', 'cifar10']")
