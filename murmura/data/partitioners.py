"""Data partitioning strategies for federated learning."""

from typing import List, Optional, Tuple
import numpy as np


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    min_samples_per_client: int = 1,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Partition data across clients using Dirichlet distribution.

    This creates non-IID data distributions where lower alpha values
    lead to more heterogeneous distributions across clients.

    Args:
        labels: Array of labels for each sample (shape: [num_samples])
        num_clients: Number of clients to partition data across
        alpha: Dirichlet concentration parameter.
               - alpha -> 0: Each client gets samples from only one class (extreme non-IID)
               - alpha -> inf: Each client gets uniform distribution (IID)
               - alpha = 1: Uniform prior over all possible distributions
               - Typical values: 0.1 (very non-IID), 0.5 (moderate), 1.0 (mild)
        min_samples_per_client: Minimum samples each client must have
        seed: Random seed for reproducibility

    Returns:
        List of lists, where partition[i] contains the dataset indices for client i
    """
    if seed is not None:
        np.random.seed(seed)

    num_samples = len(labels)
    num_classes = len(np.unique(labels))

    # Get indices for each class
    class_indices = {c: np.where(labels == c)[0] for c in np.unique(labels)}

    # Initialize client partitions
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    # For each class, distribute samples according to Dirichlet
    for class_label, indices in class_indices.items():
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Calculate number of samples per client for this class
        proportions = proportions / proportions.sum()  # Normalize
        sample_counts = (proportions * len(indices)).astype(int)

        # Handle rounding - assign remaining samples
        remaining = len(indices) - sample_counts.sum()
        if remaining > 0:
            # Add remaining to random clients
            extra_clients = np.random.choice(num_clients, remaining, replace=False)
            sample_counts[extra_clients] += 1

        # Shuffle class indices
        np.random.shuffle(indices)

        # Assign samples to clients
        current_idx = 0
        for client_id, count in enumerate(sample_counts):
            client_indices[client_id].extend(indices[current_idx : current_idx + count].tolist())
            current_idx += count

    # Ensure minimum samples per client by redistributing if needed
    _ensure_minimum_samples(client_indices, min_samples_per_client)

    # Shuffle each client's indices
    for indices in client_indices:
        np.random.shuffle(indices)

    return client_indices


def _ensure_minimum_samples(
    client_indices: List[List[int]], min_samples: int
) -> None:
    """Ensure each client has at least min_samples (in-place modification).

    Takes samples from clients with excess and gives to those with too few.
    """
    if min_samples <= 0:
        return

    # Find clients needing more samples
    deficit_clients = [
        (i, min_samples - len(indices))
        for i, indices in enumerate(client_indices)
        if len(indices) < min_samples
    ]

    if not deficit_clients:
        return

    # Find clients with surplus
    surplus_clients = [
        (i, len(indices) - min_samples)
        for i, indices in enumerate(client_indices)
        if len(indices) > min_samples
    ]

    # Redistribute
    for deficit_client, needed in deficit_clients:
        for surplus_client, available in surplus_clients:
            if available <= 0 or needed <= 0:
                continue

            take = min(needed, available)
            # Move samples
            moved = client_indices[surplus_client][-take:]
            client_indices[surplus_client] = client_indices[surplus_client][:-take]
            client_indices[deficit_client].extend(moved)

            needed -= take
            # Update surplus tracking
            surplus_clients = [
                (i, a - take) if i == surplus_client else (i, a)
                for i, a in surplus_clients
            ]


def iid_partition(
    num_samples: int,
    num_clients: int,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Partition data uniformly across clients (IID).

    Args:
        num_samples: Total number of samples in the dataset
        num_clients: Number of clients to partition data across
        seed: Random seed for reproducibility

    Returns:
        List of lists, where partition[i] contains the dataset indices for client i
    """
    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Split as evenly as possible
    splits = np.array_split(indices, num_clients)
    return [split.tolist() for split in splits]


def natural_partition(
    client_ids: np.ndarray,
    num_clients: Optional[int] = None,
) -> Tuple[List[List[int]], int]:
    """Partition data by natural client/user IDs in the dataset.

    Useful for datasets that already have user identifiers (e.g., wearable data
    collected from different subjects).

    Args:
        client_ids: Array of client/user IDs for each sample
        num_clients: Optional limit on number of clients (uses first N)

    Returns:
        Tuple of (partitions, actual_num_clients)
        - partitions: List of lists with dataset indices per client
        - actual_num_clients: Number of unique clients found/used
    """
    unique_clients = np.unique(client_ids)

    if num_clients is not None and num_clients < len(unique_clients):
        unique_clients = unique_clients[:num_clients]

    partitions = []
    for client_id in unique_clients:
        indices = np.where(client_ids == client_id)[0].tolist()
        partitions.append(indices)

    return partitions, len(unique_clients)


def combine_partitions_with_dirichlet(
    natural_partitions: List[List[int]],
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Repartition naturally partitioned data using Dirichlet distribution.

    Useful when a dataset has natural partitions (e.g., by subject) but you want
    to create a different number of clients with controlled heterogeneity.

    Args:
        natural_partitions: Existing partitions from natural_partition()
        labels: Array of labels for all samples
        num_clients: Target number of clients
        alpha: Dirichlet concentration parameter
        seed: Random seed

    Returns:
        New partitions with Dirichlet-based distribution
    """
    # Flatten all indices
    all_indices = []
    for partition in natural_partitions:
        all_indices.extend(partition)

    # Get labels for these indices
    partition_labels = labels[all_indices]

    # Apply Dirichlet partitioning
    new_partitions = dirichlet_partition(
        labels=partition_labels,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

    # Map back to original indices
    return [[all_indices[i] for i in partition] for partition in new_partitions]
