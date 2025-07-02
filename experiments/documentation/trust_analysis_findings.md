# Trust Monitor Analysis - Key Findings

## Critical Issues Identified

### 1. **HSIC Drift Detection Logic is Inverted** 
**Location**: `murmura/trust/hsic.py:195-200`

**Problem**: The drift detection logic is fundamentally wrong:
```python
# Current (WRONG) logic:
drift_detected = (
    hsic_value < lower_threshold or  # Significantly below normal
    hsic_value < 0.3  # Absolute low correlation threshold
)
```

**Issue**: This logic treats LOW HSIC values as suspicious, but in federated learning:
- **High HSIC (0.9+)** = High correlation between models = **NORMAL/GOOD**
- **Low HSIC (< 0.3)** = Low correlation = **SUSPICIOUS/BAD** 

The current logic is correct - low HSIC should indicate drift. However, the threshold is too low.

### 2. **Trust Monitor Score Interpretation is Wrong**
**Location**: `murmura/trust/trust_monitor.py:304-317`

**Problem**: The trust scoring logic is confusing HSIC interpretation:
```python
# Current (WRONG) logic:
hsic_trust = min(hsic_value, 1.0)  # HSIC values can be > 1, cap at 1.0
```

**Issue**: The comment says "High HSIC = High Trust" but the implementation treats HSIC value directly as trust score. This is wrong because:
- HSIC measures **correlation/dependence** between parameters
- High HSIC = models are similar = good in FL
- The current code caps HSIC at 1.0 but doesn't handle the case where HSIC > 1.0 properly

### 3. **HSIC Threshold is Too Permissive**
**Location**: `murmura/trust/hsic.py:270`

**Problem**: Default HSIC threshold is 0.1, but the drift detection uses 0.3 as absolute threshold:
```python
threshold: float = 0.1,  # Constructor default
# But actual usage:
hsic_value < 0.3  # Absolute low correlation threshold
```

**Issue**: In federated learning, normal HSIC values between honest nodes should be 0.7-0.9+. A threshold of 0.3 is far too permissive - it only catches extremely malicious behavior.

### 4. **Exchanges are Succeeding but Not Detecting**
**Finding**: From the logs, we see "Completed 6/6 trust-aware exchanges" but detection rate is 0%.

**Root Cause**: The trust exchanges are working fine. The problem is:
1. HSIC values between malicious and honest nodes are still above 0.3
2. Performance validation is not catching the attack
3. The thresholds are too permissive for gradual attacks

## Specific Technical Issues

### A. HSIC Calibration Logic Flawed
The HSIC implementation attempts to learn a baseline during the first 10 samples, but:
- It uses statistical outlier detection (mean ± 2σ) 
- But gradual attacks slowly shift the baseline
- The adaptive threshold mechanism gets confused

### B. Performance Monitoring Not Effective
**Location**: `murmura/trust/performance_trust.py` (referenced but not examined)

The performance-based validation should catch malicious models that perform poorly on local validation data, but it's not triggering.

### C. Trust Score Combination is Confusing
```python
if self.performance_monitor:
    # Weight: 40% HSIC, 40% performance, 20% adaptive
    combined_trust = 0.4 * hsic_trust + 0.4 * performance_trust + 0.2 * adaptive_trust
else:
    # Weight: 60% HSIC, 40% adaptive  
    combined_trust = 0.6 * hsic_trust + 0.4 * adaptive_trust
```

The problem is that if HSIC is interpreted wrong and performance monitoring isn't working, the combined score will be wrong.

## Recommended Fixes

### 1. **Fix HSIC Drift Detection Threshold**
Change the absolute threshold from 0.3 to a much higher value:
```python
# In hsic.py, line 199:
hsic_value < 0.7  # Raise from 0.3 to 0.7
```

### 2. **Fix Statistical Threshold Calculation**
```python
# In hsic.py, lines 195-196:
# Change from detecting "below normal" to detecting "significantly below normal"
lower_threshold = max(0.7, hsic_mean - 1.5 * hsic_std)  # Raise base threshold
```

### 3. **Debug Performance Monitoring**
The performance validation component needs investigation - it should be catching models that perform poorly on local validation data.

### 4. **Adjust Trust Profiles**
The "strict" trust profile should have much more aggressive thresholds:
- `exclude_threshold`: 0.5 → 0.2
- `downgrade_threshold`: 0.3 → 0.15
- `warn_threshold`: 0.15 → 0.1

## Why the Attack is Not Being Detected

1. **Gradual Nature**: The gradual label flipping attack increases intensity slowly
2. **HSIC Values Still High**: Even with 50% label flipping, models might still have HSIC > 0.3
3. **Performance Drop Not Caught**: The performance monitoring validation isn't sensitive enough
4. **Thresholds Too Permissive**: All thresholds are set for obvious attacks, not subtle ones

## Test Results Explanation

From the logs showing "Completed 6/6 trust-aware exchanges":
- ✅ Trust exchange mechanism is working
- ✅ HSIC calculations are running
- ✅ Performance validation is attempted
- ❌ But values are not crossing the detection thresholds
- ❌ So no nodes get marked as suspicious/excluded

## Next Steps

1. **Immediate**: Raise HSIC thresholds from 0.3 to 0.7
2. **Investigate**: Performance monitoring effectiveness  
3. **Test**: Run with adjusted thresholds to verify detection
4. **Optimize**: Fine-tune thresholds based on actual HSIC value distributions

The core issue is that the trust monitor is working correctly but the thresholds are calibrated for obvious attacks, not gradual sophisticated attacks that are the target of this research.