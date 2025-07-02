# Trust Monitor Implementation Details

## Code Architecture

### 1. Core Components

```
murmura/trust/
├── trust_monitor.py          # Main TrustMonitor class
├── hsic.py                   # HSIC drift detection
├── beta_threshold.py         # Adaptive thresholding
├── adaptive_trust_agent.py   # FL context integration
└── trust_config.py           # Configuration classes
```

### 2. Key Classes and Methods

#### TrustMonitor (trust_monitor.py)
```python
class TrustMonitor:
    def __init__(self, node_id, neighbors, config):
        # Single trust monitor per node
        self.node_id = node_id
        self.hsic_monitors = {}  # Dictionary of HSIC monitors, one per neighbor
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_agent = AdaptiveTrustAgent()
        
        # Initialize HSIC monitor for each neighbor
        for neighbor_id in neighbors:
            self.hsic_monitors[neighbor_id] = HSICMonitor(config.hsic_config)
        
    async def assess_update(self, neighbor_id, parameters):
        # 1. HSIC-based drift detection
        hsic_trust = self.hsic_monitors[neighbor_id].assess(parameters)
        
        # 2. Performance validation (if test data available)
        perf_trust = self.performance_monitor.validate(parameters)
        
        # 3. Adaptive weighting based on FL context
        weights = self.adaptive_agent.get_weights(fl_context)
        
        # 4. Compute final trust score
        trust_score = weights.hsic * hsic_trust + weights.perf * perf_trust
        
        # 5. Determine action
        action = self._determine_action(trust_score)
        
        return action, trust_score
```

#### HSICMonitor (hsic.py)
```python
class HSICMonitor:
    def __init__(self, config):
        self.window_size = config.window_size
        self.parameter_history = deque(maxlen=window_size)
        self.beta_threshold = BetaThreshold(config.beta_config)
        
    def compute_hsic(self, X, Y):
        # Compute RBF kernels
        K_X = self._rbf_kernel(X, X, self.gamma)
        K_Y = self._rbf_kernel(Y, Y, self.gamma)
        
        # Center kernels
        H = np.eye(n) - np.ones((n, n)) / n
        K_X_centered = H @ K_X @ H
        K_Y_centered = H @ K_Y @ H
        
        # Compute HSIC
        hsic = np.trace(K_X_centered @ K_Y_centered) / (n - 1) ** 2
        return hsic
        
    def assess_drift(self, new_params):
        # Add to history
        self.parameter_history.append(new_params)
        
        # Compute HSIC between historical and new parameters
        hsic_value = self.compute_hsic(historical_params, new_params)
        
        # Get adaptive threshold
        threshold = self.beta_threshold.get_threshold(
            self.parameter_history,
            fl_context
        )
        
        # Compute trust
        trust = min(1.0, hsic_value / threshold) if hsic_value < threshold else 1.0
        return trust
```

#### BetaThreshold (beta_threshold.py)
```python
class BetaThreshold:
    def __init__(self, config):
        self.base_percentile = config.base_percentile
        self.alpha = 1.0  # Beta distribution parameters
        self.beta = 1.0
        
    def fit_beta_distribution(self, hsic_values):
        # Method of moments estimation
        mean = np.mean(hsic_values)
        var = np.var(hsic_values)
        
        # Estimate alpha and beta
        common_factor = mean * (1 - mean) / var - 1
        self.alpha = mean * common_factor
        self.beta = (1 - mean) * common_factor
        
    def get_adaptive_percentile(self, fl_context):
        round_progress = fl_context['round'] / fl_context['total_rounds']
        
        # Adjust percentile based on FL phase
        percentile = self.base_percentile
        percentile += self.early_rounds_adjustment * (1 - round_progress)
        percentile += self.late_rounds_adjustment * round_progress
        
        return np.clip(percentile, 0.9, 0.999)
```

### 3. Attack Implementations

#### GradualLabelFlippingAttack
```python
class GradualLabelFlippingAttack:
    def poison_labels(self, features, labels):
        n_flip = int(len(labels) * self.current_intensity)
        
        # Select random indices to flip
        flip_indices = np.random.choice(len(labels), n_flip, replace=False)
        
        # Flip labels to random different classes
        for idx in flip_indices:
            original = labels[idx]
            labels[idx] = np.random.choice([i for i in range(10) if i != original])
            
        return features, labels, stats
```

#### GradualModelPoisoningAttack
```python
class GradualModelPoisoningAttack:
    def poison_data(self, features, labels):
        n_poison = int(len(labels) * self.current_poison_rate)
        
        # Select samples to poison
        poison_indices = np.random.choice(len(labels), n_poison, replace=False)
        
        # Apply backdoor trigger and change label
        for idx in poison_indices:
            features[idx] = self._apply_trigger(features[idx], self.trigger_strength)
            labels[idx] = self.target_class
            
        return features, labels, stats
        
    def _apply_trigger(self, feature, strength):
        # Add trigger pattern (e.g., pixel pattern in corner)
        poisoned = feature.copy()
        poisoned[self.trigger_mask] = strength
        return poisoned
```

### 4. Integration with Federated Learning

#### Trust-Aware Aggregation
```python
class TrustAwareTrueDecentralizedLearningProcess:
    def aggregate_neighbor_updates(self, node_id, round_num):
        updates = []
        weights = []
        
        # Single trust monitor for this node assesses all neighbor updates
        trust_monitor = self.trust_monitors[node_id]
        
        for neighbor_id, params in neighbor_updates.items():
            # Get trust assessment from this node's trust monitor
            action, trust_score = trust_monitor.assess_update(
                neighbor_id, params
            )
            
            if action != TrustAction.EXCLUDE:
                updates.append(params)
                # Apply trust-based weighting
                weight = trust_score if action == TrustAction.ACCEPT else trust_score * 0.5
                weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Weighted aggregation
        aggregated = sum(w * update for w, update in zip(weights, updates))
        return aggregated
```

### 5. Performance Optimizations

#### Efficient HSIC Computation
```python
# Use vectorized operations
def compute_pairwise_hsic(self, param_history):
    # Batch compute all pairwise HSIC values
    n = len(param_history)
    hsic_matrix = np.zeros((n, n))
    
    # Compute kernels once
    K = self._rbf_kernel(param_history, param_history)
    
    # Efficient HSIC computation
    for i in range(n):
        for j in range(i+1, n):
            hsic_matrix[i,j] = self._fast_hsic(K[i], K[j])
            
    return hsic_matrix
```

#### Memory-Efficient Parameter Storage
```python
class CompressedParameterHistory:
    def __init__(self, max_size=30, compression_ratio=0.1):
        self.pca = IncrementalPCA(n_components=int(original_dim * compression_ratio))
        self.compressed_history = deque(maxlen=max_size)
        
    def add(self, params):
        # Incrementally fit PCA and transform
        compressed = self.pca.fit_transform(params.reshape(1, -1))
        self.compressed_history.append(compressed)
```

### 6. Ray Actor Implementation

```python
@ray.remote
class TrustMonitorActor:
    def __init__(self, node_id, neighbors, config):
        # Each node gets one trust monitor that handles all neighbors
        self.trust_monitor = TrustMonitor(node_id, neighbors, config)
        
    async def assess_update(self, neighbor_id, parameters):
        # Single trust monitor assesses updates from all neighbors
        return await self.trust_monitor.assess_update(neighbor_id, parameters)
        
    def get_trust_report(self):
        # Returns trust status for all monitored neighbors
        return self.trust_monitor.get_comprehensive_report()
```

## Testing and Validation

### Unit Tests
```python
def test_hsic_computation():
    # Test HSIC with known distributions
    X = np.random.normal(0, 1, (100, 50))
    Y = X + np.random.normal(0, 0.1, (100, 50))  # Correlated
    
    hsic_high = compute_hsic(X, Y)
    assert hsic_high > 0.8  # High correlation
    
    Z = np.random.normal(0, 1, (100, 50))  # Independent
    hsic_low = compute_hsic(X, Z)
    assert hsic_low < 0.3  # Low correlation
```

### Integration Tests
```python
def test_gradual_attack_detection():
    # Setup trust monitor
    trust_monitor = TrustMonitor(config)
    
    # Simulate gradual attack
    for round in range(15):
        if round < 5:
            # Honest behavior
            trust_score = trust_monitor.assess_update(honest_params)
            assert trust_score > 0.7
        else:
            # Attack behavior
            trust_score = trust_monitor.assess_update(poisoned_params)
            if round > 8:  # Should detect by moderate phase
                assert trust_score < 0.5
```

## Configuration Examples

### Default Configuration
```python
trust_config = TrustMonitoringConfig(
    hsic_config=HSICConfig(
        window_size=30,
        kernel_type="rbf",
        gamma=0.1,
        reduce_dim=True,
        target_dim=50
    ),
    beta_config=BetaThresholdConfig(
        base_percentile=0.98,
        early_rounds_adjustment=-0.05,
        late_rounds_adjustment=0.01,
        min_observations=8,
        learning_rate=0.5
    ),
    policy_config=TrustPolicyConfig(
        warn_threshold=0.7,
        downgrade_threshold=0.5,
        exclude_threshold=0.2,
        min_samples_for_action=8
    )
)
```

### Edge Device Configuration
```python
edge_trust_config = TrustMonitoringConfig(
    hsic_config=HSICConfig(
        window_size=20,        # Smaller window
        target_dim=30,         # More aggressive reduction
        kernel_type="linear"   # Faster computation
    ),
    performance_validation=False,  # No test data
    trust_report_frequency=5       # Less frequent reports
)
```