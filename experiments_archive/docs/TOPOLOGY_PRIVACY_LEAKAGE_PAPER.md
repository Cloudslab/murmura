# Topology Privacy Leakage in Distributed Learning: A Comprehensive Analysis

This document serves as the complete technical reference for our academic paper on topology-based privacy leakage in federated and decentralized learning systems. It contains all methodologies, algorithms, configurations, and experimental details needed to reproduce our work.

## Abstract Summary

We present a comprehensive study demonstrating how network topology knowledge enables data distribution inference attacks against distributed learning systems, even with differential privacy protections. Through systematic evaluation across 5-30 node networks, multiple topologies (star, ring, complete, line), and varying privacy levels (ε = 1.0 to 16.0), we show that structural information leaks persist despite state-of-the-art privacy mechanisms.

---

## 1. Attack Modeling and Assumptions

### 1.1 Threat Model

**Adversary Capabilities:**
- **Knowledge of Network Topology**: Complete knowledge of the communication graph structure, including node connectivity patterns, degrees, and positions
- **Passive Observation**: Can observe communication patterns (frequency, timing, metadata) but cannot modify messages
- **Parameter Access**: Has access to model parameter updates during aggregation (realistic in many FL scenarios)
- **No Direct Data Access**: Cannot access raw training data at any node

**Adversary Goals:**
- **Data Distribution Inference**: Determine which nodes contain specific data characteristics (e.g., sensitive classes, demographic groups)
- **Group Membership**: Identify nodes belonging to sensitive groups or having similar data distributions
- **Class Correlation**: Infer relationships between topology position and data class distributions

### 1.2 Attack Surface Analysis

Our attack surface consists of three primary information leakage channels:

1. **Communication Pattern Leakage**
   - Frequency of parameter exchanges between nodes
   - Timing patterns in decentralized communication
   - Message size variations (though we assume size uniformity)

2. **Parameter Magnitude Leakage**
   - Statistical properties of parameter updates (norms, means, standard deviations)
   - Evolution patterns of parameter magnitudes over training rounds
   - Systematic differences between nodes with different data distributions

3. **Structural Position Leakage**
   - Correlation between topology position and data characteristics
   - Influence of node degree and centrality on observed parameters
   - Patterns specific to topology type (ring ordering, star centrality, etc.)

### 1.3 Assumptions and Constraints

**Realistic Assumptions:**
- Semi-honest participants (follow protocol but may observe/infer)
- Network topology is determined by organizational/geographical constraints
- Data distributions naturally correlate with organizational structure
- Standard aggregation protocols (FedAvg, GossipAvg) without specialized privacy mechanisms

**Privacy Mechanism Assumptions:**
- Standard differential privacy implementation via Opacus
- Gaussian noise addition with proper calibration
- RDP (Rényi Differential Privacy) accounting methodology
- No additional privacy-preserving aggregation techniques

---

## 2. Experimental Methodology and Architecture

### 2.1 Two-Phase Experimental Design

#### Phase 1: Comprehensive Baseline Analysis
**Objectives:**
- Establish attack effectiveness across all topology-privacy combinations
- Measure baseline vulnerability without sampling effects
- Identify most vulnerable configurations

**Coverage:**
- 2 datasets × 3 attack strategies × 2 FL types × multiple topologies × 5 node counts × 5 DP levels
- Total: ~450 unique configurations with full participation

#### Phase 2: Sampling Effects Analysis
**Objectives:**
- Evaluate impact of realistic FL sampling on attack success
- Test privacy amplification by subsampling
- Validate results under practical deployment constraints

**Sampling Scenarios:**
- **Moderate Sampling**: 50% client participation, 80% data sampling
- **Strong Sampling**: 30% client participation, 60% data sampling  
- **Very Strong Sampling**: 20% client participation, 50% data sampling

### 2.2 Experimental Framework Architecture

```
Murmura Framework
├── Learning Processes
│   ├── FederatedLearningProcess (centralized, star/complete topologies)
│   └── DecentralizedLearningProcess (peer-to-peer, ring/line/complete)
├── Attack-Oriented Data Partitioning
│   ├── SensitiveGroupPartitioner
│   ├── TopologyCorrelatedPartitioner
│   └── ImbalancedSensitivePartitioner
├── Privacy Framework
│   ├── DPConfig (comprehensive ε/δ management)
│   ├── PrivacyAccountant (RDP-based tracking)
│   └── Opacus Integration (automated noise calibration)
├── Attack Implementation
│   ├── CommunicationPatternAttack
│   ├── ParameterMagnitudeAttack
│   └── TopologyStructureAttack
└── Distributed Execution
    ├── Ray-based multi-node coordination
    ├── AWS G5.2xlarge cluster management
    └── Comprehensive logging/visualization
```

### 2.3 Datasets and Models

#### MNIST Dataset
- **Size**: 60,000 training samples, 10,000 test samples
- **Classes**: 10 digit classes (0-9)
- **Model**: Convolutional Neural Network (CNN)
  - Conv2D(32, 3x3) → ReLU → Conv2D(64, 3x3) → ReLU → MaxPool(2x2)
  - Dropout(0.25) → Flatten → Dense(128) → ReLU → Dropout(0.5) → Dense(10)
- **Optimal Nodes**: 10 (matches number of classes for topology correlation)

#### HAM10000 Skin Lesion Dataset
- **Size**: ~10,000 dermatoscopic images
- **Classes**: 7 skin lesion types (nv, mel, bkl, bcc, akiec, vasc, df)
- **Model**: Modified ResNet-18 for medical imaging
  - Pre-trained backbone with medical domain adaptation
  - Custom classifier head for 7-class classification
- **Optimal Nodes**: 7 (matches number of classes)

### 2.4 Attack Strategy Implementations

#### Strategy 1: Sensitive Groups (sensitive_groups)
**Concept**: Specific topology positions contain sensitive demographic groups

**Implementation**:
- Partition classes into "sensitive" and "non-sensitive" groups
- Assign sensitive groups to specific topology positions
- For MNIST: {0,1,2,3,4} vs {5,6,7,8,9} mapped to node positions
- For HAM10000: {mel, bcc, akiec} (malignant/concerning) vs {nv, bkl, vasc, df} (benign)

**Attack Vector**: Communication pattern analysis reveals group membership through interaction frequency

#### Strategy 2: Topology Correlated (topology_correlated)
**Concept**: Data distribution directly correlates with topology position

**Implementation**:
- **Ring Topology**: Sequential nodes get sequential classes (node 0→class 0, node 1→class 1, etc.)
- **Star Topology**: Center node gets mixed data, leaves get specialized classes
- **Line Topology**: Similar to ring with end effects
- Correlation strength parameter controls perfect vs partial correlation

**Attack Vector**: Topology structure attack exploits position-class relationships

#### Strategy 3: Imbalanced Sensitive (imbalanced_sensitive)
**Concept**: Severe class imbalances at specific topology positions

**Implementation**:
- Designate specific nodes as "rare class holders"
- 90% of rare class samples go to designated nodes
- 10% distributed to other nodes for realism
- Creates detectable imbalance signatures

**Attack Vector**: Parameter magnitude differences reveal imbalanced distributions

---

## 3. Key Algorithms

### 3.1 Aggregation Algorithms

Our experimental evaluation uses two aggregation algorithms corresponding to the two learning paradigms:

#### 3.1.1 FedAvg (Federated Averaging)

**Coordination Mode**: Centralized  
**Compatible Topologies**: Star, Complete  
**Use Case**: Standard federated learning with central server

```python
def federated_averaging(parameters_list, weights=None):
    """
    Standard FedAvg algorithm for centralized federated learning
    """
    if len(parameters_list) == 1:
        return parameters_list[0]
    
    # Initialize equal weights if not provided
    if weights is None:
        weights = [1.0 / len(parameters_list)] * len(parameters_list)
    else:
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    aggregated_params = {}
    
    for param_name in parameters_list[0].keys():
        if is_integer_parameter(param_name):
            # Handle tracking parameters (use maximum)
            aggregated_params[param_name] = max(
                params[param_name] for params in parameters_list
            )
        else:
            # Weighted average for trainable parameters
            weighted_sum = zero_like(parameters_list[0][param_name])
            
            for i, params in enumerate(parameters_list):
                weighted_sum += weights[i] * params[param_name]
            
            aggregated_params[param_name] = weighted_sum
    
    return aggregated_params
```

#### 3.1.2 GossipAvg (Decentralized Averaging)

**Coordination Mode**: Decentralized  
**Compatible Topologies**: Ring, Complete, Line  
**Use Case**: Peer-to-peer learning without central coordination

```python
def gossip_averaging(local_params, neighbor_params_list, mixing_parameter=0.5):
    """
    Gossip averaging for decentralized learning
    Each node exchanges with neighbors and updates locally
    """
    if not neighbor_params_list:
        return local_params  # No neighbors to aggregate with
    
    # Combine local parameters with neighbor parameters
    all_params = [local_params] + neighbor_params_list
    
    # Equal weighting across all participants (local + neighbors)
    weights = [1.0 / len(all_params)] * len(all_params)
    
    aggregated_params = {}
    
    for param_name in local_params.keys():
        if is_integer_parameter(param_name):
            # Handle tracking parameters (use maximum)
            aggregated_params[param_name] = max(
                params[param_name] for params in all_params
            )
        else:
            # Weighted average with all participants
            weighted_sum = zero_like(local_params[param_name])
            
            for i, params in enumerate(all_params):
                weighted_sum += weights[i] * params[param_name]
            
            aggregated_params[param_name] = weighted_sum
    
    return aggregated_params

def decentralized_gossip_round(nodes, topology_adjacency_matrix):
    """
    Complete gossip round for decentralized topology
    """
    new_parameters = {}
    
    for node_i in nodes:
        # Find neighbors from topology
        neighbors = get_neighbors(node_i, topology_adjacency_matrix)
        
        # Collect neighbor parameters
        neighbor_params = [nodes[j].parameters for j in neighbors]
        
        # Perform local gossip averaging
        new_parameters[node_i] = gossip_averaging(
            local_params=nodes[node_i].parameters,
            neighbor_params_list=neighbor_params,
            mixing_parameter=0.5
        )
    
    # Update all nodes simultaneously (synchronous update)
    for node_i in nodes:
        nodes[node_i].parameters = new_parameters[node_i]
    
    return nodes
```

**Algorithm Selection by Topology and FL Type:**
```python
aggregation_mapping = {
    "federated": {
        "star": "fedavg",
        "complete": "fedavg"
    },
    "decentralized": {
        "ring": "gossip_avg",
        "complete": "gossip_avg", 
        "line": "gossip_avg"
    }
}

# Experimental usage in paper_experiments.py:
if config['fl_type'] == "federated":
    aggregation_strategy = "fedavg"
else:  # decentralized
    aggregation_strategy = "gossip_avg"
```

---

### 3.2 Attack Algorithm Implementations

#### 3.2.1 Communication Pattern Attack Algorithm

```python
def communication_pattern_attack(communications_df):
    """
    Infer node similarities from communication patterns
    """
    # Build communication frequency matrix
    nodes = get_unique_nodes(communications_df)
    comm_matrix = zeros(len(nodes), len(nodes))
    
    for communication in communications_df:
        src_idx = node_to_index[communication.source]
        tgt_idx = node_to_index[communication.target] 
        comm_matrix[src_idx][tgt_idx] += 1
    
    # Extract features: outgoing + incoming patterns
    features = concatenate([comm_matrix, comm_matrix.T], axis=1)
    
    # Cluster nodes based on communication behavior
    clusters = KMeans(n_clusters=2).fit_predict(features)
    
    # Calculate attack success metric
    coherence = max_cluster_size / total_nodes
    
    return {
        'node_clusters': clusters,
        'attack_success_metric': coherence,
        'communication_matrix': comm_matrix
    }
```

#### 3.2.2 Parameter Magnitude Attack Algorithm

```python
def parameter_magnitude_attack(parameter_updates_df):
    """
    Infer data characteristics from parameter update statistics
    """
    node_statistics = {}
    
    for node_id in unique_nodes:
        node_data = filter_by_node(parameter_updates_df, node_id)
        
        # Extract statistical features
        norms = extract_parameter_norms(node_data)
        node_statistics[node_id] = {
            'norm_mean': mean(norms),
            'norm_std': std(norms),
            'norm_trend': calculate_linear_trend(norms),
            'convergence_rate': analyze_convergence(norms),
            'stability': std(norms[-3:])  # Last 3 rounds
        }
    
    # Create feature matrix for clustering
    features = []
    for stats in node_statistics.values():
        features.append([
            stats['norm_mean'], stats['norm_std'], 
            stats['norm_trend'], stats['convergence_rate']
        ])
    
    # Normalize and cluster
    normalized_features = normalize(features)
    clusters = KMeans(n_clusters=2).fit_predict(normalized_features)
    
    # Multi-metric separability score
    separability = calculate_separability_score(normalized_features)
    
    return {
        'node_statistics': node_statistics,
        'clusters': clusters,
        'attack_success_metric': separability
    }

def calculate_separability_score(features):
    """
    Multi-metric approach for robust separability measurement
    """
    if len(features) < 4:
        # Small sample: use normalized range
        norms = features[:, 0]  # First feature (norm_mean)
        normalized_range = (max(norms) - min(norms)) / (4 * std(norms))
        return min(normalized_range, 1.0)
    else:
        # Large sample: use silhouette score
        kmeans = KMeans(n_clusters=min(len(features)//2, 3))
        cluster_labels = kmeans.fit_predict(features)
        silhouette = silhouette_score(features, cluster_labels)
        return (silhouette + 1) / 2  # Convert [-1,1] to [0,1]
```

#### 3.2.3 Topology Structure Attack Algorithm

```python
def topology_structure_attack(topology_df, parameter_updates_df):
    """
    Exploit topology structure to predict data distributions
    """
    # Extract topology features
    topology_features = {}
    for node in topology_df:
        topology_features[node.id] = {
            'degree': node.degree,
            'position': node.id,  # Position in ordering
            'is_central': 1 if node.degree > 2 else 0,
            'neighbor_sum': sum(node.connected_nodes)
        }
    
    # Extract node characteristics from parameters
    node_characteristics = {}
    for node_id in unique_nodes:
        node_data = filter_by_node(parameter_updates_df, node_id)
        norms = extract_parameter_norms(node_data)
        
        node_characteristics[node_id] = {
            'avg_norm': mean(norms),
            'norm_variability': std(norms),
            'initial_norm': norms[0],
            'final_norm': norms[-1]
        }
    
    # Test correlations between topology and characteristics
    correlations = {}
    common_nodes = intersect(topology_features.keys(), 
                           node_characteristics.keys())
    
    positions = [topology_features[n]['position'] for n in common_nodes]
    degrees = [topology_features[n]['degree'] for n in common_nodes]
    avg_norms = [node_characteristics[n]['avg_norm'] for n in common_nodes]
    
    correlations['position_vs_norm'] = pearson_correlation(positions, avg_norms)
    correlations['degree_vs_norm'] = pearson_correlation(degrees, avg_norms)
    
    # Attack success = maximum absolute correlation
    max_correlation = max(abs(corr) for corr in correlations.values())
    
    return {
        'correlations': correlations,
        'attack_success_metric': max_correlation,
        'topology_features': topology_features
    }
```

### 3.3 Differential Privacy Implementation

```python
class DifferentialPrivacyWrapper:
    """
    Comprehensive DP implementation with Opacus integration
    """
    def __init__(self, config: DPConfig):
        self.config = config
        self.accountant = PrivacyAccountant(config)
        
    def add_noise(self, parameters, round_num, client_id):
        """Add calibrated Gaussian noise to parameters"""
        
        # Get noise multiplier (auto-tuned for target epsilon)
        noise_multiplier = self.get_noise_multiplier()
        
        # Add Gaussian noise to each parameter
        noisy_params = {}
        for key, value in parameters.items():
            if 'num_batches_tracked' not in key:  # Skip non-trainable params
                noise_scale = noise_multiplier * self.config.max_grad_norm
                noise = np.random.normal(0, noise_scale, value.shape)
                noisy_params[key] = value + noise
            else:
                noisy_params[key] = value
        
        # Record privacy expenditure
        self.accountant.record_training_round(
            client_id=client_id,
            noise_multiplier=noise_multiplier,
            sample_rate=self.config.sample_rate or 1.0,
            steps=1,  # One step per round in our setup
            round_number=round_num
        )
        
        return noisy_params
    
    def get_noise_multiplier(self):
        """Get optimally calibrated noise multiplier"""
        if self.config.auto_tune_noise:
            return self.accountant.suggest_optimal_noise(
                sample_rate=self.config.sample_rate or 1.0,
                epochs=self.config.epochs or 1,
                dataset_size=60000,  # Will be set dynamically
                target_epsilon=self.config.target_epsilon
            )
        return self.config.noise_multiplier or 1.0
    
    def apply_subsampling_amplification(self, base_epsilon):
        """Apply privacy amplification by subsampling"""
        if not self.config.use_amplification_by_subsampling:
            return base_epsilon
            
        client_rate = self.config.client_sampling_rate or 1.0
        data_rate = self.config.data_sampling_rate or 1.0
        amplification_factor = client_rate * data_rate
        
        # Simplified amplification (real implementation uses complex bounds)
        amplified_epsilon = base_epsilon * amplification_factor
        
        return amplified_epsilon
```

---

## 4. Infrastructure and Configuration Details

### 4.1 AWS Infrastructure Configuration

**Cluster Specification:**
- **Instance Type**: AWS G5.2xlarge
- **Quantity**: 10 machines
- **Per-Machine Resources**:
  - 8 vCPU cores (AMD EPYC 7R32)
  - 1 NVIDIA A10G GPU (24GB GPU memory)
  - 32 GiB system RAM
  - 450 GB NVMe SSD storage
  - Up to 25 Gbps network performance

**Ray Cluster Configuration:**
```python
ray_cluster_config = {
    "address": "auto",  # Auto-discovery for multi-node
    "num_nodes": 10,
    "head_node_cpu": 8,
    "head_node_memory": 32000,  # MB
    "worker_node_cpu": 8,
    "worker_node_memory": 32000,
    "placement_strategy": "SPREAD",  # Distribute across nodes
    "enable_placement_groups": True
}

resource_config = {
    "actors_per_node": 3,  # Up to 30 total virtual clients
    "cpus_per_actor": 2,
    "gpus_per_actor": 0.1,  # Shared GPU usage
    "memory_per_actor": 4000  # MB per actor
}
```

### 4.2 Privacy Accounting Methodology

**RDP (Rényi Differential Privacy) Approach:**
- **Accounting Framework**: Opacus RDPAccountant
- **Alpha Values**: Default Opacus sequence [1.25, 1.5, 1.75, 2., 2.25, ..., 64]
- **Composition**: Automatic composition across training rounds
- **Noise Calibration**: Automated to achieve target ε with optimal utility

**Privacy Parameters:**
```python
privacy_levels = {
    "no_dp": {"enabled": False, "epsilon": None},
    "weak_dp": {"enabled": True, "epsilon": 16.0, "delta": 1e-5},
    "medium_dp": {"enabled": True, "epsilon": 8.0, "delta": 1e-5}, 
    "strong_dp": {"enabled": True, "epsilon": 4.0, "delta": 1e-5},
    "very_strong_dp": {"enabled": True, "epsilon": 1.0, "delta": 1e-6}
}

# Gradient clipping and noise
max_grad_norm = 1.0  # L2 norm clipping threshold
noise_mechanism = "gaussian"  # Gaussian noise addition
secure_mode = False  # Standard PyTorch RNG (not cryptographically secure)
```

**Subsampling Amplification (Phase 2):**
```python
amplification_scenarios = {
    "moderate": {"client_rate": 0.5, "data_rate": 0.8},
    "strong": {"client_rate": 0.3, "data_rate": 0.6},
    "very_strong": {"client_rate": 0.2, "data_rate": 0.5}
}

# Privacy amplification calculation
effective_sample_rate = client_rate * data_rate * base_sample_rate
amplified_epsilon = compute_rdp_epsilon(
    noise_multiplier=sigma,
    sample_rate=effective_sample_rate,
    steps=total_steps,
    delta=target_delta
)
```

### 4.3 Experimental Parameters

**Training Configuration:**
```python
training_params = {
    "rounds": 3,  # Federated learning rounds (kept low for large-scale experiments)
    "local_epochs": 1,  # Local training epochs per round
    "batch_size": 32,  # Local batch size
    "learning_rate": 0.001,  # Learning rate
    "optimizer": "adam",  # Optimizer choice
    "loss_function": "cross_entropy"
}

# Dataset-specific adjustments
mnist_params = {
    "image_size": 28,
    "input_channels": 1,
    "num_classes": 10
}

ham10000_params = {
    "image_size": 128,  # Dermatoscopic images
    "input_channels": 3,
    "num_classes": 7
}
```

**Network Topologies:**
```python
topology_configs = {
    "star": {
        "compatible_fl_types": ["federated"],
        "aggregation_strategies": ["fedavg", "trimmed_mean"],
        "description": "Central server with leaf clients"
    },
    "ring": {
        "compatible_fl_types": ["decentralized"], 
        "aggregation_strategies": ["gossip_avg"],
        "description": "Circular communication pattern"
    },
    "complete": {
        "compatible_fl_types": ["federated", "decentralized"],
        "aggregation_strategies": ["fedavg", "gossip_avg"],
        "description": "Fully connected graph"
    },
    "line": {
        "compatible_fl_types": ["decentralized"],
        "aggregation_strategies": ["gossip_avg"], 
        "description": "Linear chain topology"
    }
}
```

### 4.4 Attack Evaluation Metrics

**Success Threshold**: 0.3 (attacks with confidence > 30% considered successful)

**Primary Evaluation Metrics:**
1. **Communication Pattern Attack**: Cluster coherence ratio
2. **Parameter Magnitude Attack**: Multi-metric separability score combining:
   - Silhouette score (for samples ≥ 4)
   - Normalized range metric (for smaller samples)
   - Feature variance score
3. **Topology Structure Attack**: Maximum absolute correlation coefficient

**Metric Extraction and Comparison Pipeline:**
Our evaluation focuses specifically on **Parameter Magnitude Attack** metrics for primary analysis, as implemented in `rerun_attacks.py`:

```python
def compare_attack_metrics(old_results, new_results):
    """Extract and compare Parameter Magnitude Attack success metrics"""
    comparisons = []
    
    for old_result in old_results:
        exp_name = old_result["experiment_name"]
        new_result = find_matching_experiment(new_results, exp_name)
        
        if new_result and new_result["status"] == "success":
            # Extract Parameter Magnitude Attack metrics specifically
            old_mag_attack = extract_attack_metric(old_result, "Parameter Magnitude Attack")
            new_mag_attack = extract_attack_metric(new_result, "Parameter Magnitude Attack")
            
            if old_mag_attack is not None and new_mag_attack is not None:
                comparisons.append({
                    "experiment": exp_name,
                    "old_metric": old_mag_attack,
                    "new_metric": new_mag_attack,
                    "improvement": new_mag_attack - old_mag_attack,
                    "improvement_ratio": new_mag_attack / old_mag_attack
                })
    
    return comparisons

def extract_attack_metric(result, attack_name):
    """Extract specific attack success metric from results"""
    for attack in result["attack_results"]["attack_results"]:
        if attack["attack_name"] == attack_name:
            return attack["attack_success_metric"]
    return None
```

**Phase 2 Sampling Impact Analysis:**
```python
def analyze_phase2_sampling_effects(attack_result, config):
    """Analyze how subsampling affects attack effectiveness"""
    analysis = {
        "sampling_scenario": config.get("sampling_scenario", "unknown"),
        "client_sampling_rate": config.get("client_sampling_rate", 1.0),
        "data_sampling_rate": config.get("data_sampling_rate", 1.0),
        "sampling_impact_metrics": {}
    }
    
    if attack_result["status"] == "success":
        for attack in attack_result.get("attack_results", []):
            attack_name = attack.get("attack_name", "")
            success_metric = attack.get("attack_success_metric", 0.0)
            
            # Calculate participation factor (combined sampling effect)
            participation_factor = (config.get("client_sampling_rate", 1.0) * 
                                  config.get("data_sampling_rate", 1.0))
            expected_reduction = 1.0 - participation_factor
            
            analysis["sampling_impact_metrics"][attack_name] = {
                "success_metric": success_metric,
                "participation_factor": participation_factor,
                "expected_reduction": expected_reduction,
                "relative_effectiveness": success_metric / max(participation_factor, 0.1)
            }
    
    return analysis
```

**Cross-Phase Comparison Methodology:**
```python
def compare_phase1_vs_phase2(phase1_results, phase2_results):
    """Compare baseline vs sampling-affected attack success"""
    
    # Group Phase 1 results by configuration (excluding sampling parameters)
    phase1_by_config = {}
    for result in phase1_results:
        if result["status"] == "success":
            config = result["config"]
            key = f"{config['dataset']}_{config['fl_type']}_{config['topology']}" \
                  f"_{config['node_count']}_{config['dp_setting']['name']}" \
                  f"_{config['attack_strategy']}"
            phase1_by_config[key] = result
    
    # Compare with Phase 2 results
    comparisons = []
    for phase2_result in phase2_results:
        if phase2_result["status"] == "success":
            config = phase2_result["config"]
            key = create_config_key(config)  # Same key generation
            
            if key in phase1_by_config:
                phase1_result = phase1_by_config[key]
                
                # Extract Parameter Magnitude Attack metrics for comparison
                phase1_metric = extract_attack_metric(phase1_result, "Parameter Magnitude Attack")
                phase2_metric = extract_attack_metric(phase2_result, "Parameter Magnitude Attack")
                
                comparisons.append({
                    "config_key": key,
                    "sampling_scenario": config.get("sampling_scenario", "unknown"),
                    "client_sampling_rate": config.get("client_sampling_rate", 1.0),
                    "data_sampling_rate": config.get("data_sampling_rate", 1.0),
                    "phase1_metric": phase1_metric,
                    "phase2_metric": phase2_metric,
                    "metric_reduction": phase1_metric - phase2_metric,
                    "relative_reduction": (phase1_metric - phase2_metric) / max(phase1_metric, 0.001)
                })
    
    return comparisons
```

**Key Evaluation Focus:**
- **Primary Metric**: Parameter Magnitude Attack success scores (most reliable across configurations)
- **Comparative Analysis**: Before/after metric improvements and cross-phase reductions
- **Sampling Impact**: Quantified effectiveness reduction under realistic participation rates
- **Statistical Validation**: Average improvements, improvement ratios, and relative reductions

### 4.5 Data Collection and Visualization

**Comprehensive Logging:**
- **Training Events**: Parameter updates, communication logs, convergence metrics
- **Network Topology**: Adjacency matrices, node attributes, connection patterns
- **Attack Results**: Individual attack outputs, success metrics, confidence scores
- **Privacy Tracking**: Epsilon/delta expenditure, noise multiplier values, amplification factors

**Output Files per Experiment:**
```
experiment_name/
├── training_data_metrics.csv          # Training accuracy/loss over rounds
├── training_data_communications.csv   # All inter-node communications
├── training_data_parameter_updates.csv # Parameter statistics per round
├── training_data_topology.csv         # Network structure information
├── training_data_adjacency_matrix.csv # Topology adjacency matrix
├── training_data_node_attributes.csv  # Node-specific metadata
└── summary_plot.png                   # Training convergence visualization
```

**Automated Analysis Pipeline:**
1. **Training Execution**: Ray-distributed training with comprehensive logging
2. **Attack Execution**: Automated attack runs on collected data
3. **Results Aggregation**: JSON export with full experimental metadata
4. **Statistical Analysis**: Pandas-based analysis with LaTeX table generation
5. **Visualization**: Matplotlib plots for attack success patterns

---

## 5. Experimental Validity and Reproducibility

### 5.1 Statistical Rigor

**Sample Sizes:**
- Phase 1: ~450 unique configurations with deterministic results
- Phase 2: ~150 configurations with sampling variance analysis
- Multiple runs with different random seeds for sampling experiments

**Success Criteria:**
- Conservative threshold (30%) to avoid false positives
- Multi-metric evaluation to ensure robust attack detection
- Cross-validation across different attack types

### 5.2 Reproducibility Guarantees

**Deterministic Components:**
- Fixed random seeds for data partitioning
- Identical model architectures across experiments
- Consistent training hyperparameters
- Standardized privacy parameter ranges

**Environmental Control:**
- Identical AWS instance types across all experiments
- Ray cluster configuration standardization
- Docker containerization for software dependencies
- Automated experiment orchestration to minimize human error

**Code and Data Availability:**
- Complete codebase with detailed documentation
- Experimental configuration files for all scenarios
- Automated reproduction scripts
- Comprehensive logging for debugging and verification

---

## 6. Expected Contributions and Impact

### 6.1 Novel Contributions

1. **First Systematic Study** of topology-based privacy leakage across multiple FL architectures
2. **Comprehensive Attack Framework** with three complementary attack vectors
3. **Differential Privacy Evaluation** under realistic subsampling conditions
4. **Scalability Analysis** across varying network sizes (5-30 nodes)
5. **Multi-Dataset Validation** on both synthetic (MNIST) and real-world (medical) data

### 6.2 Practical Impact

**For FL System Designers:**
- Guidelines for topology-aware privacy preservation
- Quantified privacy-utility tradeoffs across different architectures
- Recommendations for secure topology design

**For Privacy Researchers:**
- Novel attack vectors beyond traditional reconstruction attacks
- Demonstration of DP limitations in structured federated settings
- Framework for evaluating topology-specific privacy threats

**For Practitioners:**
- Practical privacy parameter recommendations
- Infrastructure configuration guidelines for secure deployments
- Risk assessment methodology for topology-dependent threats

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

**Scope Constraints:**
- Limited to passive adversaries (no active attacks)
- Assumes known topology structure (realistic in many organizations)
- Focus on classification tasks (no regression or other ML paradigms)

**Technical Limitations:**
- Simplified privacy amplification bounds (conservative estimates)
- Limited attack sophistication (room for more advanced inference)
- Standard aggregation protocols (no specialized privacy-preserving mechanisms)

### 7.2 Future Research Directions

1. **Advanced Attack Development**: More sophisticated inference attacks using graph neural networks
2. **Defense Mechanisms**: Topology-aware privacy-preserving aggregation protocols
3. **Dynamic Topology Analysis**: Privacy implications of adaptive/changing network structures
4. **Multi-Modal Learning**: Extension to federated learning with diverse data types
5. **Large-Scale Validation**: Experiments with hundreds of participants in realistic organizational settings

---

## Conclusion

This comprehensive methodology provides a rigorous framework for evaluating topology-based privacy leakage in distributed learning systems. Our systematic approach, combining realistic threat models, comprehensive attack implementations, and large-scale empirical evaluation, establishes new insights into the fundamental privacy challenges posed by network structure in federated learning deployments.

The combination of our novel attack framework, extensive experimental coverage, and practical infrastructure configuration provides both immediate actionable insights for practitioners and a solid foundation for future privacy research in distributed machine learning.