"""
Resource-specific events for tracking communication, computation, and sampling metrics
"""

from typing import Dict, List, Optional, Any
from murmura.visualization.training_event import TrainingEvent


class ResourceTrackingEvent(TrainingEvent):
    """Base class for resource tracking events"""
    
    def __init__(self, round_num: int, resource_type: str):
        super().__init__(round_num, f"resource_{resource_type}")
        self.resource_type = resource_type


class CommunicationEvent(ResourceTrackingEvent):
    """Event for tracking communication metrics"""
    
    def __init__(
        self,
        round_num: int,
        source_nodes: List[int],
        target_nodes: List[int],
        bytes_transferred: int,
        communication_time: float,
        message_type: str = "parameters",  # parameters, gradients, metrics
        direction: str = "bidirectional"  # upload, download, bidirectional
    ):
        super().__init__(round_num, "communication")
        self.source_nodes = source_nodes
        self.target_nodes = target_nodes
        self.bytes_transferred = bytes_transferred
        self.communication_time = communication_time
        self.message_type = message_type
        self.direction = direction
        self.bandwidth_mbps = (bytes_transferred * 8 / 1e6) / communication_time if communication_time > 0 else 0


class ComputationTimeEvent(ResourceTrackingEvent):
    """Event for tracking computation time metrics"""
    
    def __init__(
        self,
        round_num: int,
        node_id: int,
        computation_type: str,  # training, evaluation, aggregation
        time_taken: float,
        samples_processed: Optional[int] = None,
        batches_processed: Optional[int] = None,
        epochs_completed: Optional[int] = None
    ):
        super().__init__(round_num, "computation_time")
        self.node_id = node_id
        self.computation_type = computation_type
        self.time_taken = time_taken
        self.samples_processed = samples_processed
        self.batches_processed = batches_processed
        self.epochs_completed = epochs_completed
        # Calculate throughput if applicable
        self.throughput_samples_per_sec = samples_processed / time_taken if samples_processed and time_taken > 0 else None


class MemoryUsageEvent(ResourceTrackingEvent):
    """Event for tracking memory usage"""
    
    def __init__(
        self,
        round_num: int,
        node_id: int,
        memory_mb: float,
        memory_type: str = "total",  # total, model, data, gradients
        peak_memory_mb: Optional[float] = None
    ):
        super().__init__(round_num, "memory_usage")
        self.node_id = node_id
        self.memory_mb = memory_mb
        self.memory_type = memory_type
        self.peak_memory_mb = peak_memory_mb or memory_mb


class SamplingEvent(ResourceTrackingEvent):
    """Event for tracking client and data sampling"""
    
    def __init__(
        self,
        round_num: int,
        total_clients: int,
        sampled_clients: List[int],
        client_sampling_rate: float,
        data_sampling_rates: Optional[Dict[int, float]] = None,  # per-client data sampling
        total_data_points: Optional[int] = None,
        sampled_data_points: Optional[int] = None
    ):
        super().__init__(round_num, "sampling")
        self.total_clients = total_clients
        self.sampled_clients = sampled_clients
        self.num_sampled_clients = len(sampled_clients)
        self.client_sampling_rate = client_sampling_rate
        self.actual_client_rate = self.num_sampled_clients / total_clients if total_clients > 0 else 0
        self.data_sampling_rates = data_sampling_rates or {}
        self.total_data_points = total_data_points
        self.sampled_data_points = sampled_data_points
        self.actual_data_rate = sampled_data_points / total_data_points if total_data_points and sampled_data_points else None


class AggregationCostEvent(ResourceTrackingEvent):
    """Event for tracking aggregation costs"""
    
    def __init__(
        self,
        round_num: int,
        aggregation_time: float,
        num_clients_aggregated: int,
        parameters_size_bytes: int,
        aggregation_type: str,  # fedavg, trimmed_mean, gossip_avg
        cpu_usage_percent: Optional[float] = None,
        memory_usage_mb: Optional[float] = None
    ):
        super().__init__(round_num, "aggregation_cost")
        self.aggregation_time = aggregation_time
        self.num_clients_aggregated = num_clients_aggregated
        self.parameters_size_bytes = parameters_size_bytes
        self.aggregation_type = aggregation_type
        self.cpu_usage_percent = cpu_usage_percent
        self.memory_usage_mb = memory_usage_mb
        # Calculate aggregation throughput
        self.throughput_mb_per_sec = (parameters_size_bytes * num_clients_aggregated / 1e6) / aggregation_time if aggregation_time > 0 else 0


class PrivacyAmplificationEvent(ResourceTrackingEvent):
    """Event for tracking privacy amplification from subsampling"""
    
    def __init__(
        self,
        round_num: int,
        base_epsilon: float,
        effective_epsilon: float,
        client_sampling_rate: float,
        data_sampling_rate: float,
        amplification_factor: float
    ):
        super().__init__(round_num, "privacy_amplification")
        self.base_epsilon = base_epsilon
        self.effective_epsilon = effective_epsilon
        self.client_sampling_rate = client_sampling_rate
        self.data_sampling_rate = data_sampling_rate
        self.amplification_factor = amplification_factor
        self.privacy_savings_percent = (1 - effective_epsilon / base_epsilon) * 100 if base_epsilon > 0 else 0


class ResourceSummaryEvent(ResourceTrackingEvent):
    """Event for round-wise resource usage summary"""
    
    def __init__(
        self,
        round_num: int,
        total_communication_bytes: int,
        total_computation_time: float,
        total_clients_involved: int,
        sampled_clients_count: int,
        round_completion_time: float,
        resource_efficiency_score: Optional[float] = None
    ):
        super().__init__(round_num, "resource_summary")
        self.total_communication_bytes = total_communication_bytes
        self.total_computation_time = total_computation_time
        self.total_clients_involved = total_clients_involved
        self.sampled_clients_count = sampled_clients_count
        self.round_completion_time = round_completion_time
        self.resource_efficiency_score = resource_efficiency_score
        # Calculate savings
        self.communication_savings_percent = (1 - sampled_clients_count / total_clients_involved) * 100 if total_clients_involved > 0 else 0
        self.time_per_client = round_completion_time / sampled_clients_count if sampled_clients_count > 0 else 0