"""
Attack-oriented partitioners for testing topology-based privacy leaks.
These partitioners create known sensitive distributions to validate attacks.
"""
from typing import List, Optional, Dict, cast
import numpy as np
from murmura.data_processing.partitioner import Partitioner
from murmura.data_processing.dataset import MDataset


class SensitiveGroupPartitioner(Partitioner):
    """
    Creates partitions where specific nodes contain sensitive subgroups.
    Used to test if topology structure can reveal sensitive group membership.
    
    For MNIST: Groups certain digits (e.g., 0,1 vs 8,9) to specific topology positions
    This tests if ring position correlates with data characteristics.
    """
    
    def __init__(
        self,
        num_partitions: int,
        sensitive_groups: Dict[str, List[int]],  # {"group_name": [class_labels]}
        topology_assignment: Dict[str, List[int]],  # {"group_name": [node_ids]}
        seed: Optional[int] = 42
    ):
        super().__init__(num_partitions, seed)
        self.sensitive_groups = sensitive_groups
        self.topology_assignment = topology_assignment
        self._validate_assignment()
    
    def _validate_assignment(self):
        """Validate that all nodes are assigned and no overlaps exist."""
        all_nodes = set(range(self.num_partitions))
        assigned_nodes = set()
        
        for group, nodes in self.topology_assignment.items():
            if group not in self.sensitive_groups:
                raise ValueError(f"Group {group} not defined in sensitive_groups")
            assigned_nodes.update(nodes)
        
        if assigned_nodes != all_nodes:
            raise ValueError("All nodes must be assigned to exactly one group")
    
    def partition(
        self, dataset: MDataset, split_name: str, partition_by: Optional[str] = None
    ) -> None:
        """Create partitions based on sensitive group assignments."""
        partition_by = partition_by or "label"
        split_dataset = dataset.get_split(split_name)
        targets = np.array(split_dataset[partition_by])
        
        self.partitions = {i: [] for i in range(self.num_partitions)}
        
        # Group samples by class
        class_indices = {}
        for cls in np.unique(targets):
            class_indices[cls] = np.where(targets == cls)[0].tolist()
        
        # Distribute classes to assigned nodes
        for group_name, class_list in self.sensitive_groups.items():
            node_list = self.topology_assignment[group_name]
            
            # Collect all indices for this group
            group_indices = []
            for cls in class_list:
                if cls in class_indices:
                    group_indices.extend(class_indices[cls])
            
            # Shuffle and distribute among assigned nodes
            if group_indices:
                self.rng.shuffle(group_indices)
                chunks = np.array_split(group_indices, len(node_list))
                
                for node_id, chunk in zip(node_list, chunks):
                    self.partitions[node_id].extend(chunk.tolist())
        
        dataset.add_partitions(split_name, cast(Dict[int, List[int]], self.partitions))


class TopologyCorrelatedPartitioner(Partitioner):
    """
    Creates data distributions that correlate with topology position.
    Tests if sequential communication patterns leak positional information.
    
    For ring topology: Sequential nodes get sequential classes (0→1→2→3→4)
    For star topology: Center gets mixed data, leaves get specialized classes
    """
    
    def __init__(
        self,
        num_partitions: int,
        topology_type: str = "ring",  # "ring", "star", "line"
        correlation_strength: float = 0.8,  # 0.5 = random, 1.0 = perfect correlation
        seed: Optional[int] = 42
    ):
        super().__init__(num_partitions, seed)
        self.topology_type = topology_type
        self.correlation_strength = correlation_strength
    
    def partition(
        self, dataset: MDataset, split_name: str, partition_by: Optional[str] = None
    ) -> None:
        """Create topology-correlated partitions."""
        partition_by = partition_by or "label"
        split_dataset = dataset.get_split(split_name)
        targets = np.array(split_dataset[partition_by])
        unique_classes = np.unique(targets)
        
        self.partitions = {i: [] for i in range(self.num_partitions)}
        
        if self.topology_type == "ring":
            self._create_ring_correlation(targets, unique_classes)
        elif self.topology_type == "star":
            self._create_star_correlation(targets, unique_classes)
        elif self.topology_type == "line":
            self._create_line_correlation(targets, unique_classes)
        else:
            raise ValueError(f"Unsupported topology: {self.topology_type}")
        
        dataset.add_partitions(split_name, cast(Dict[int, List[int]], self.partitions))
    
    def _create_ring_correlation(self, targets, unique_classes):
        """Ring: Sequential nodes get sequential classes.
        
        Fixed to handle cases where num_classes != num_partitions properly.
        Each node should get exactly one primary class when possible.
        """
        # First, assign primary classes to nodes (one class per node when possible)
        for node_id in range(min(self.num_partitions, len(unique_classes))):
            cls = unique_classes[node_id]
            cls_indices = np.where(targets == cls)[0]
            
            # Determine how much goes to primary vs other nodes
            n_primary = int(len(cls_indices) * self.correlation_strength)
            n_secondary = len(cls_indices) - n_primary
            
            self.rng.shuffle(cls_indices)
            
            # Assign primary samples to this node
            self.partitions[node_id].extend(cls_indices[:n_primary].tolist())
            
            # Distribute remainder to other nodes
            if n_secondary > 0:
                other_nodes = [i for i in range(self.num_partitions) if i != node_id]
                secondary_chunks = np.array_split(cls_indices[n_primary:], len(other_nodes))
                
                for other_node, chunk in zip(other_nodes, secondary_chunks):
                    if len(chunk) > 0:
                        self.partitions[other_node].extend(chunk.tolist())
        
        # Handle remaining classes if more classes than nodes
        if len(unique_classes) > self.num_partitions:
            for cls_idx in range(self.num_partitions, len(unique_classes)):
                cls = unique_classes[cls_idx]
                cls_indices = np.where(targets == cls)[0]
                
                # Distribute these extra classes evenly across all nodes
                self.rng.shuffle(cls_indices)
                chunks = np.array_split(cls_indices, self.num_partitions)
                for node_id, chunk in enumerate(chunks):
                    if len(chunk) > 0:
                        self.partitions[node_id].extend(chunk.tolist())
    
    def _create_star_correlation(self, targets, unique_classes):
        """Star: Center node (0) gets mixed data, leaves get specialized."""
        center_node = 0
        leaf_nodes = list(range(1, self.num_partitions))
        
        for cls_idx, cls in enumerate(unique_classes):
            cls_indices = np.where(targets == cls)[0]
            
            if cls_idx < len(leaf_nodes):
                # Assign primarily to corresponding leaf
                leaf_node = leaf_nodes[cls_idx]
                n_leaf = int(len(cls_indices) * self.correlation_strength)
                n_center = len(cls_indices) - n_leaf
                
                self.rng.shuffle(cls_indices)
                self.partitions[leaf_node].extend(cls_indices[:n_leaf].tolist())
                self.partitions[center_node].extend(cls_indices[n_leaf:].tolist())
            else:
                # Distribute remaining classes to center
                self.partitions[center_node].extend(cls_indices.tolist())
    
    def _create_line_correlation(self, targets, unique_classes):
        """Line: Similar to ring but with end effects."""
        self._create_ring_correlation(targets, unique_classes)  # Same logic for now


class ImbalancedSensitivePartitioner(Partitioner):
    """
    Creates severe class imbalances at specific topology positions.
    Tests if communication frequency/parameter magnitude leaks class distribution.
    """
    
    def __init__(
        self,
        num_partitions: int,
        rare_class_nodes: List[int],  # Nodes that get rare classes
        rare_classes: List[int],      # Which classes are rare
        rarity_factor: float = 0.1,   # Fraction of rare class samples
        seed: Optional[int] = 42
    ):
        super().__init__(num_partitions, seed)
        self.rare_class_nodes = rare_class_nodes
        self.rare_classes = rare_classes
        self.rarity_factor = rarity_factor
    
    def partition(
        self, dataset: MDataset, split_name: str, partition_by: Optional[str] = None
    ) -> None:
        """Create imbalanced partitions with rare classes at specific nodes."""
        partition_by = partition_by or "label"
        split_dataset = dataset.get_split(split_name)
        targets = np.array(split_dataset[partition_by])
        
        self.partitions = {i: [] for i in range(self.num_partitions)}
        
        # Separate rare and common classes
        rare_indices = {}
        common_indices = {}
        
        for cls in np.unique(targets):
            cls_indices = np.where(targets == cls)[0]
            if cls in self.rare_classes:
                rare_indices[cls] = cls_indices
            else:
                common_indices[cls] = cls_indices
        
        # Distribute rare classes to create true imbalance
        for cls, indices in rare_indices.items():
            # Most samples go to rare nodes (1 - rarity_factor), few to others
            n_for_rare_nodes = int(len(indices) * (1 - self.rarity_factor))
            n_for_other_nodes = len(indices) - n_for_rare_nodes
            
            self.rng.shuffle(indices)
            
            # Most rare samples go to designated rare_class_nodes
            if n_for_rare_nodes > 0:
                rare_chunks = np.array_split(indices[:n_for_rare_nodes], len(self.rare_class_nodes))
                for node, chunk in zip(self.rare_class_nodes, rare_chunks):
                    self.partitions[node].extend(chunk.tolist())
            
            # Few samples go to non-rare nodes (to create imbalance)
            if n_for_other_nodes > 0:
                non_rare_nodes = [i for i in range(self.num_partitions) if i not in self.rare_class_nodes]
                if non_rare_nodes:  # Only distribute if there are non-rare nodes
                    remaining_chunks = np.array_split(indices[n_for_rare_nodes:], len(non_rare_nodes))
                    for node, chunk in zip(non_rare_nodes, remaining_chunks):
                        self.partitions[node].extend(chunk.tolist())
                else:
                    # If all nodes are rare nodes, distribute remainder among them too
                    remaining_chunks = np.array_split(indices[n_for_rare_nodes:], len(self.rare_class_nodes))
                    for node, chunk in zip(self.rare_class_nodes, remaining_chunks):
                        self.partitions[node].extend(chunk.tolist())
        
        # Distribute common classes primarily to non-rare nodes
        non_rare_nodes = [i for i in range(self.num_partitions) if i not in self.rare_class_nodes]
        
        for cls, indices in common_indices.items():
            self.rng.shuffle(indices)
            
            if non_rare_nodes:
                # Most common class samples go to non-rare nodes
                n_for_non_rare = int(len(indices) * 0.9)  # 90% to non-rare nodes
                n_for_rare = len(indices) - n_for_non_rare
                
                # Distribute to non-rare nodes
                if n_for_non_rare > 0:
                    non_rare_chunks = np.array_split(indices[:n_for_non_rare], len(non_rare_nodes))
                    for node, chunk in zip(non_rare_nodes, non_rare_chunks):
                        self.partitions[node].extend(chunk.tolist())
                
                # Small amount to rare nodes (to maintain some diversity)
                if n_for_rare > 0 and self.rare_class_nodes:
                    rare_chunks = np.array_split(indices[n_for_non_rare:], len(self.rare_class_nodes))
                    for node, chunk in zip(self.rare_class_nodes, rare_chunks):
                        self.partitions[node].extend(chunk.tolist())
            else:
                # If no non-rare nodes, distribute evenly
                chunks = np.array_split(indices, self.num_partitions)
                for node, chunk in enumerate(chunks):
                    self.partitions[node].extend(chunk.tolist())
        
        dataset.add_partitions(split_name, cast(Dict[int, List[int]], self.partitions))