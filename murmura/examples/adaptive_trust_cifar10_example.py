#!/usr/bin/env python3
"""
Adaptive Trust CIFAR-10 Example with Beta Distribution-based Thresholding.

This example demonstrates the adaptive trust monitoring system with CIFAR-10,
a more complex dataset than MNIST, showing trust system scalability.
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, Any, Optional

import numpy as np
import ray

# Core FL components
from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.models.cifar_models import CIFAR10Model, SimpleCIFAR10Model
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.node.resource_config import RayClusterConfig, ResourceConfig
from murmura.orchestration.learning_process.trust_aware_true_decentralized_learning_process import (
    TrustAwareTrueDecentralizedLearningProcess,
)
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.model.pytorch_model import TorchModelWrapper

# Enhanced trust components
from murmura.trust.trust_config import TrustMonitoringConfig, HSICConfig, TrustPolicyConfig
from murmura.trust.beta_threshold import BetaThresholdConfig

# Attack components
from murmura.attacks.gradual_label_flipping import create_gradual_attack_config
from murmura.attacks.gradual_model_poisoning import create_backdoor_config


def setup_logging(log_level: str = "INFO") -> None:
    """Set up comprehensive logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/adaptive_trust_cifar10.log"),
        ],
    )
    
    # Set specific loggers to reduce noise
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def create_adaptive_trust_config(
    profile: str = "default",
    use_beta_threshold: bool = True,
    topology_type: str = "ring",
    enable_ensemble: bool = False,
    num_classes: int = 10,
    custom_config: Optional[Dict[str, Any]] = None
) -> TrustMonitoringConfig:
    """
    Create adaptive trust configuration for CIFAR-10.
    
    CIFAR-10 models are larger and more complex than MNIST, so we adjust
    the trust parameters accordingly.
    
    Args:
        profile: Trust profile (permissive, default, strict)
        use_beta_threshold: Whether to use Beta distribution-based thresholding
        topology_type: Network topology (ring, complete, line) for optimization
        custom_config: Custom configuration overrides
    """
    
    # Beta threshold configuration optimized for CIFAR-10
    if use_beta_threshold:
        if profile == "permissive":
            beta_config = BetaThresholdConfig(
                base_percentile=0.995,          # Even higher for complex models
                early_rounds_adjustment=-0.03,  # More permissive early
                late_rounds_adjustment=0.01,    # Slightly stricter late
                min_observations=15,            # More observations for stability
                learning_rate=0.2,              # Conservative learning
            )
        elif profile == "strict":
            beta_config = BetaThresholdConfig(
                base_percentile=0.97,           # Lower percentile
                early_rounds_adjustment=-0.02,  # Less permissive early
                late_rounds_adjustment=0.02,    # More strict late
                min_observations=8,             # Fewer observations needed
                learning_rate=0.6,              # Faster learning
            )
        else:  # default
            beta_config = BetaThresholdConfig(
                base_percentile=0.985,          # High percentile for complex FL
                early_rounds_adjustment=-0.04,  # More permissive early
                late_rounds_adjustment=0.015,   # Slightly stricter late
                min_observations=12,            # Balanced observations
                learning_rate=0.4,              # Moderate learning
            )
    else:
        beta_config = None
    
    # HSIC configuration optimized for CIFAR-10 (larger parameter space)
    hsic_config = HSICConfig(
        window_size=25,             # Smaller window for faster adaptation
        kernel_type="rbf",
        gamma=0.05,                 # Adjusted for larger parameter space
        threshold=0.05,             # Will be overridden by Beta threshold
        alpha=0.95,                 # Higher alpha for more stable estimates
        reduce_dim=True,
        target_dim=100,             # Appropriate for CIFAR-10 model size
        calibration_rounds=5,
        baseline_percentile=96.0,   # Higher baseline for complex models
    )
    
    # Trust policy configuration for CIFAR-10 (topology-aware)
    if profile == "permissive":
        trust_policy = TrustPolicyConfig(
            warn_threshold=0.25,
            downgrade_threshold=0.45,
            exclude_threshold=0.65,
            min_samples_for_action=15,  # More samples for complex models
            weight_reduction_factor=0.7,
        )
    elif profile == "strict":
        trust_policy = TrustPolicyConfig(
            warn_threshold=0.1,
            downgrade_threshold=0.25,
            exclude_threshold=0.45,
            min_samples_for_action=8,
            weight_reduction_factor=0.3,
        )
    else:  # default
        trust_policy = TrustPolicyConfig(
            warn_threshold=0.15,
            downgrade_threshold=0.35,
            exclude_threshold=0.55,
            min_samples_for_action=12,
            weight_reduction_factor=0.5,
        )
    
    # Adjust trust policy for topology characteristics (CIFAR-10 specific)
    if topology_type == "complete":
        # Complete graph: more neighbors, can be more strict even with complex models
        trust_policy.exclude_threshold = max(0.15, trust_policy.exclude_threshold - 0.1)
        trust_policy.min_samples_for_action = max(5, trust_policy.min_samples_for_action - 3)
    elif topology_type == "line":
        # Line topology: fewer neighbors, be more permissive (especially important for CIFAR-10)
        trust_policy.exclude_threshold = min(0.8, trust_policy.exclude_threshold + 0.15)
        trust_policy.min_samples_for_action = min(20, trust_policy.min_samples_for_action + 5)
    # Ring topology uses default values
    
    # Apply custom overrides
    if custom_config:
        if "hsic" in custom_config:
            for key, value in custom_config["hsic"].items():
                setattr(hsic_config, key, value)
        if "trust_policy" in custom_config:
            for key, value in custom_config["trust_policy"].items():
                setattr(trust_policy, key, value)
    
    config = TrustMonitoringConfig(
        enabled=True,
        hsic_config=hsic_config,
        trust_policy_config=trust_policy,
        log_trust_metrics=True,
        trust_report_interval=3,    # Less frequent due to complexity
        enable_ensemble_detection=enable_ensemble,
        num_classes=num_classes,
    )
    
    # Store Beta config for later use
    config.beta_threshold_config = beta_config
    
    return config


def create_attack_config(
    attack_type: str = "none",
    malicious_fraction: float = 0.0,
    attack_intensity: str = "moderate",
    stealth_level: str = "medium",
    target_class: int = 0
) -> Optional[Dict[str, Any]]:
    """Create attack configuration for different attack types on CIFAR-10."""
    
    if attack_type == "none" or malicious_fraction <= 0:
        return None
    
    if attack_type == "gradual_label_flipping" or attack_type == "label_flipping":
        # Use the gradual label flipping attack configuration
        attack_config_obj = create_gradual_attack_config(
            dataset_name="cifar10",
            attack_intensity=attack_intensity,
            stealth_level=stealth_level
        )
        
        # Convert AttackConfig object to dictionary and add extra fields
        config = {
            "attack_config": attack_config_obj,
            "malicious_fraction": malicious_fraction,
            "attack_type": "label_flipping",
            "attack_intensity": attack_intensity,
            "stealth_level": stealth_level
        }
        
        return config
    
    elif attack_type == "model_poisoning" or attack_type == "backdoor":
        # Use the gradual model poisoning (backdoor) attack configuration
        attack_config_obj = create_backdoor_config(
            dataset_name="cifar10",
            attack_intensity=attack_intensity,
            stealth_level=stealth_level,
            target_class=target_class
        )
        
        # Convert BackdoorConfig object to dictionary and add extra fields
        config = {
            "attack_config": attack_config_obj,
            "malicious_fraction": malicious_fraction,
            "attack_type": "model_poisoning",
            "attack_intensity": attack_intensity,
            "stealth_level": stealth_level,
            "target_class": target_class,
            "input_shape": (32, 32, 3)  # CIFAR-10 shape
        }
        
        return config
    
    else:
        # Unknown attack type
        raise ValueError(f"Unknown attack type: {attack_type}. Supported types: none, label_flipping, model_poisoning")


def run_adaptive_trust_cifar10(
    num_actors: int = 6,
    num_rounds: int = 20,
    topology_type: str = "ring",
    trust_profile: str = "default",
    use_beta_threshold: bool = True,
    model_type: str = "standard",
    attack_config: Optional[Dict[str, Any]] = None,
    output_dir: str = "adaptive_trust_cifar10_results",
    log_level: str = "INFO",
    enable_ensemble: bool = False
) -> Dict[str, Any]:
    """
    Run complete CIFAR-10 experiment with adaptive trust monitoring.
    
    Args:
        num_actors: Number of federated learning actors
        num_rounds: Number of FL rounds (CIFAR-10 needs more than MNIST)
        topology_type: Decentralized network topology (ring, complete, line)
        trust_profile: Trust monitoring profile (permissive, default, strict)
        use_beta_threshold: Whether to use Beta distribution thresholding
        model_type: Model architecture (simple, standard, resnet)
        attack_config: Attack configuration dictionary (if any)
        output_dir: Directory to save results
        log_level: Logging level
        
    Returns:
        Dictionary with experiment results
    """
    
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger("adaptive_trust_cifar10")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("ADAPTIVE TRUST CIFAR-10 EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Actors: {num_actors}, Rounds: {num_rounds}")
    logger.info(f"Topology: {topology_type}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Trust Profile: {trust_profile}")
    logger.info(f"Beta Thresholding: {use_beta_threshold}")
    if attack_config:
        attack_info = f"{attack_config.get('attack_type', 'None')} ({attack_config.get('attack_intensity', 'moderate')})"
        logger.info(f"Attack: {attack_info}, Malicious: {attack_config.get('malicious_fraction', 0)*100:.0f}%")
    else:
        logger.info(f"Attack: None")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Initialize Ray
        if not ray.is_initialized():
            logger.info("Initializing Ray...")
            ray.init(ignore_reinit_error=True)
        
        logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
        
        # Create CIFAR-10 model
        logger.info(f"Creating CIFAR-10 {model_type} model...")
        if model_type == "simple":
            base_model = SimpleCIFAR10Model()
        elif model_type == "resnet":
            from murmura.models.cifar_models import ResNetCIFAR10
            base_model = ResNetCIFAR10()
        else:  # standard
            base_model = CIFAR10Model()
            
        model = TorchModelWrapper(
            model=base_model,
            input_shape=(3, 32, 32),  # CIFAR-10 RGB images
        )
        
        # Load CIFAR-10 dataset
        logger.info("Loading CIFAR-10 dataset...")
        dataset = MDataset.load(
            DatasetSource.HUGGING_FACE,
            dataset_name="cifar10",
            split=None,
        )
        logger.info(f"Dataset splits available: {dataset.available_splits}")
        
        # Create data partitioner
        logger.info("Creating IID data partitions...")
        temp_config = OrchestrationConfig(
            num_actors=num_actors,
            partition_strategy="iid",  # Use IID for honest baseline
            feature_columns=["img"],   # CIFAR-10 uses 'img' not 'image'
            label_column="label",
            dataset_name="cifar10",
        )
        
        partitioner = PartitionerFactory.create(temp_config)
        partitioner.partition(dataset, "train")
        logger.info(f"Created {num_actors} data partitions")
        
        # Configure network topology (only decentralized topologies)
        topology_types = {
            "ring": TopologyType.RING,
            "complete": TopologyType.COMPLETE,
            "line": TopologyType.LINE,
        }
        
        # Validate topology choice
        if topology_type not in topology_types:
            logger.warning(f"Unknown topology '{topology_type}', defaulting to 'ring'")
            topology_type = "ring"
        
        selected_topology = topology_types[topology_type]
        topology_config = TopologyConfig(
            topology_type=selected_topology,
        )
        
        logger.info(f"Using {topology_type.upper()} topology with {selected_topology}")
        
        # Configure aggregation
        aggregation_config = AggregationConfig(
            strategy_type=AggregationStrategyType.GOSSIP_AVG,
            params={"mixing_parameter": 0.5},
        )
        
        # Configure adaptive trust monitoring for CIFAR-10
        ensemble_msg = " with ensemble detection" if enable_ensemble else ""
        logger.info(f"Configuring adaptive trust monitoring for CIFAR-10 (profile: {trust_profile}, topology: {topology_type}){ensemble_msg}...")
        trust_config = create_adaptive_trust_config(
            profile=trust_profile,
            use_beta_threshold=use_beta_threshold,
            topology_type=topology_type,
            enable_ensemble=enable_ensemble,
            num_classes=10  # CIFAR-10 has 10 classes
        )
        
        # Configure Ray resources
        ray_cluster_config = RayClusterConfig(
            namespace="adaptive_trust_cifar10",
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    "MURMURA_LOG_LEVEL": log_level,
                }
            },
        )
        
        resource_config = ResourceConfig(
            placement_strategy="spread",
        )
        
        # Create complete orchestration config
        orchestration_config = OrchestrationConfig(
            num_actors=num_actors,
            topology=topology_config,
            aggregation=aggregation_config,
            ray_cluster=ray_cluster_config,
            resources=resource_config,
            dataset_name="cifar10",
            partition_strategy="iid",
            feature_columns=["img"],
            label_column="label",
            rounds=num_rounds,
            epochs=3,               # More local epochs for CIFAR-10
            batch_size=64,          # Larger batch size for CIFAR-10
            learning_rate=0.001,    # Standard learning rate
            trust_monitoring=trust_config,
        )
        
        # Create trust-aware learning process
        logger.info("Initializing trust-aware true decentralized learning process...")
        learning_process = TrustAwareTrueDecentralizedLearningProcess(
            config=orchestration_config,
            dataset=dataset,
            model=model,
        )
        
        # Initialize with attack config if provided
        learning_process.initialize(
            num_actors=num_actors,
            topology_config=topology_config,
            aggregation_config=aggregation_config,
            partitioner=partitioner,
            attack_config=attack_config,
        )
        
        # Trust monitors will be automatically configured during execution
        # ==== PATTERN ANALYSIS MODE: Skip performance monitoring setup ====
        # During pattern analysis, we focus on parameter collection rather than performance metrics
        logger.info("Trust monitors will be configured for PATTERN ANALYSIS mode during execution")
        
        # ==== COMMENTED OUT: PERFORMANCE MONITORING SETUP ====
        # # Configure trust monitors with CIFAR-10 context
        # logger.info("Configuring trust monitors for CIFAR-10...")
        # if hasattr(learning_process, 'trust_monitors') and learning_process.trust_monitors:
        #     # Prepare test data for performance monitoring (subset of test split)
        #     test_dataset = dataset.get_split("test")
        #     test_features, test_labels = learning_process._prepare_test_data(
        #         test_dataset, orchestration_config.feature_columns, orchestration_config.label_column
        #     )
        #     
        #     # Use a smaller subset for performance monitoring (CIFAR-10 is more complex)
        #     perf_test_size = min(300, len(test_features))  # Smaller subset due to complexity
        #     indices = np.random.choice(len(test_features), perf_test_size, replace=False)
        #     perf_test_features = test_features[indices]
        #     perf_test_labels = test_labels[indices]
        #     
        #     logger.info(f"Prepared performance test data: {perf_test_size} samples")
        #     
        #     for node_id, trust_monitor_ref in learning_process.trust_monitors.items():
        #         # Set FL context for CIFAR-10
        #         ray.get(trust_monitor_ref.set_fl_context.remote(
        #             total_rounds=num_rounds,
        #             current_accuracy=0.1,  # Initial accuracy (random = 10% for 10 classes)
        #             topology=topology_type
        #         ))
        #         
        #         # Configure Beta thresholding if available
        #         if hasattr(trust_config, 'beta_threshold_config') and trust_config.beta_threshold_config:
        #             ray.get(trust_monitor_ref.configure_beta_threshold.remote(
        #                 trust_config.beta_threshold_config.to_dict()
        #             ))
        #         
        #         # Set test data for performance monitoring
        #         ray.get(trust_monitor_ref.set_test_data.remote(
        #             perf_test_features, perf_test_labels
        #         ))
        #             
        #     logger.info(f"Configured {len(learning_process.trust_monitors)} trust monitors for CIFAR-10")
        # else:
        #     logger.warning("No trust monitors available for configuration")
        
        # Execute federated learning
        logger.info("=" * 60)
        logger.info("STARTING CIFAR-10 FEDERATED LEARNING WITH ADAPTIVE TRUST")
        logger.info("=" * 60)
        
        results = learning_process.execute()
        
        # Get final trust report from results
        final_trust_report = results.get("final_trust_report", {})
        
        # Compile comprehensive results
        comprehensive_results = {
            "experiment_config": {
                "num_actors": num_actors,
                "num_rounds": num_rounds,
                "topology_type": topology_type,
                "model_type": model_type,
                "trust_profile": trust_profile,
                "use_beta_threshold": use_beta_threshold,
                "attack_type": attack_config.get("attack_type", "none") if attack_config else "none",
                "dataset": "cifar10",
                "timestamp": time.time(),
            },
            "fl_results": results,
            "trust_analysis": {
                "final_trust_report": final_trust_report,
                "trust_enabled": trust_config.enabled,
                "beta_thresholding": use_beta_threshold,
            },
            "performance_metrics": {
                "total_time": time.time() - start_time,
                "accuracy_improvement": results.get("accuracy_improvement", 0),
                "initial_accuracy": results.get("initial_metrics", {}).get("consensus_accuracy", 0),
                "final_accuracy": results.get("final_metrics", {}).get("consensus_accuracy", 0),
            }
        }
        
        # Analyze trust results
        if final_trust_report and "global_stats" in final_trust_report:
            stats = final_trust_report["global_stats"]
            comprehensive_results["trust_analysis"].update({
                "total_excluded": stats.get("total_excluded", 0),
                "total_downgraded": stats.get("total_downgraded", 0),
                "avg_trust_score": stats.get("avg_trust_score", 1.0),
                "false_positive_rate": stats.get("total_excluded", 0) / (num_actors * (num_actors - 1)) if num_actors > 1 else 0,
            })
        
        # Log comprehensive results
        logger.info("=" * 60)
        logger.info("CIFAR-10 EXPERIMENT RESULTS")
        logger.info("=" * 60)
        
        # FL Performance
        initial_acc = results.get('initial_metrics', {}).get('consensus_accuracy', 0)
        final_acc = results.get('final_metrics', {}).get('consensus_accuracy', 0)
        accuracy_improvement = results.get('accuracy_improvement', final_acc - initial_acc)
        
        logger.info(f"Initial Accuracy: {initial_acc:.4f}")
        logger.info(f"Final Accuracy: {final_acc:.4f}")
        logger.info(f"Accuracy Improvement: {accuracy_improvement:.4f}")
        logger.info(f"Total Training Time: {time.time() - start_time:.2f}s")
        
        # Trust Analysis
        trust_stats = comprehensive_results["trust_analysis"]
        logger.info(f"\nTrust Monitoring Results:")
        logger.info(f"  Excluded Nodes: {trust_stats.get('total_excluded', 0)}/{num_actors}")
        logger.info(f"  Downgraded Nodes: {trust_stats.get('total_downgraded', 0)}/{num_actors}")
        logger.info(f"  Average Trust Score: {trust_stats.get('avg_trust_score', 1.0):.4f}")
        logger.info(f"  False Positive Rate: {trust_stats.get('false_positive_rate', 0):.3f}")
        logger.info(f"  Beta Thresholding: {'Enabled' if use_beta_threshold else 'Disabled'}")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"cifar10_trust_results_{int(time.time())}.json")
        with open(results_file, "w") as f:
            def convert_numpy(obj):
                if hasattr(obj, "item"):
                    return obj.item()
                elif hasattr(obj, "tolist"):
                    return obj.tolist()
                return obj
            
            json.dump(comprehensive_results, f, indent=2, default=convert_numpy)
        
        logger.info(f"\nDetailed results saved to: {results_file}")
        
        # Cleanup
        try:
            learning_process.cleanup()
        except AttributeError:
            pass
        
        # Success indicator
        success_msg = "✅ CIFAR-10 EXPERIMENT COMPLETED SUCCESSFULLY"
        if trust_stats.get("total_excluded", 0) == 0:
            success_msg += " - ZERO FALSE POSITIVES!"
        
        logger.info("=" * 60)
        logger.info(success_msg)
        logger.info("=" * 60)
        
        return comprehensive_results
        
    except Exception as e:
        logger.error(f"CIFAR-10 experiment failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Ensure Ray cleanup
        if ray.is_initialized():
            try:
                ray.shutdown()
            except:
                pass


def main():
    """Main function with command-line interface for CIFAR-10 experiments."""
    
    parser = argparse.ArgumentParser(
        description="Adaptive Trust CIFAR-10 Federated Learning Example"
    )
    
    # Core experiment parameters
    parser.add_argument(
        "--num_actors", type=int, default=6,
        help="Number of federated learning actors (default: 6)"
    )
    parser.add_argument(
        "--num_rounds", type=int, default=20,
        help="Number of FL rounds (default: 20, more than MNIST due to complexity)"
    )
    parser.add_argument(
        "--topology", choices=["ring", "complete", "line"], default="ring",
        help="Decentralized network topology: ring (2 neighbors), complete (all neighbors), line (1-2 neighbors) (default: ring)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_type", choices=["simple", "standard", "resnet"], default="standard",
        help="Model architecture (default: standard)"
    )
    
    # Trust configuration
    parser.add_argument(
        "--trust_profile", choices=["permissive", "default", "strict"], default="default",
        help="Trust monitoring profile (default: default)"
    )
    parser.add_argument(
        "--disable_beta", action="store_true",
        help="Disable Beta distribution thresholding"
    )
    parser.add_argument(
        "--enable_ensemble", action="store_true",
        help="Enable ensemble trust detection (combines multiple signals)"
    )
    
    # Attack configuration
    parser.add_argument(
        "--attack_type", choices=["none", "gradual_label_flipping", "label_flipping", "model_poisoning", "backdoor"], 
        default="none", help="Type of attack to simulate (default: none)"
    )
    parser.add_argument(
        "--malicious_fraction", type=float, default=0.0,
        help="Fraction of malicious actors (default: 0.0)"
    )
    parser.add_argument(
        "--attack_intensity", choices=["low", "moderate", "high"], default="moderate",
        help="Attack intensity level for gradual attacks (default: moderate)"
    )
    parser.add_argument(
        "--stealth_level", choices=["low", "medium", "high"], default="medium",
        help="Attack stealth/evasion level for gradual attacks (default: medium)"
    )
    parser.add_argument(
        "--target_class", type=int, default=0,
        help="Target class for model poisoning attacks (default: 0)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="adaptive_trust_cifar10_results",
        help="Output directory (default: adaptive_trust_cifar10_results)"
    )
    parser.add_argument(
        "--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # Experimental scenarios
    parser.add_argument(
        "--run_baseline", action="store_true",
        help="Run honest-only baseline experiment"
    )
    parser.add_argument(
        "--run_comparison", action="store_true",
        help="Run comparison between Beta and manual thresholding"
    )
    
    args = parser.parse_args()
    
    # Create attack config
    attack_config = create_attack_config(
        attack_type=args.attack_type,
        malicious_fraction=args.malicious_fraction,
        attack_intensity=args.attack_intensity,
        stealth_level=args.stealth_level,
        target_class=args.target_class
    )
    
    if args.run_baseline:
        # Run honest-only baseline
        print("Running CIFAR-10 honest-only baseline experiment...")
        run_adaptive_trust_cifar10(
            num_actors=args.num_actors,
            num_rounds=args.num_rounds,
            topology_type=args.topology,
            trust_profile=args.trust_profile,
            use_beta_threshold=not args.disable_beta,
            model_type=args.model_type,
            attack_config=None,  # No attacks
            output_dir=os.path.join(args.output_dir, "baseline"),
            log_level=args.log_level,
            enable_ensemble=args.enable_ensemble
        )
    elif args.run_comparison:
        # Run comparison between Beta and manual thresholding
        print("Running CIFAR-10 Beta vs Manual thresholding comparison...")
        
        # Beta threshold experiment
        print("\n1. Running CIFAR-10 with Beta distribution thresholding...")
        run_adaptive_trust_cifar10(
            num_actors=args.num_actors,
            num_rounds=args.num_rounds,
            topology_type=args.topology,
            trust_profile=args.trust_profile,
            use_beta_threshold=True,
            model_type=args.model_type,
            attack_config=attack_config,
            output_dir=os.path.join(args.output_dir, "beta_threshold"),
            log_level=args.log_level,
            enable_ensemble=args.enable_ensemble
        )
        
        # Manual threshold experiment
        print("\n2. Running CIFAR-10 with manual thresholding...")
        run_adaptive_trust_cifar10(
            num_actors=args.num_actors,
            num_rounds=args.num_rounds,
            topology_type=args.topology,
            trust_profile=args.trust_profile,
            use_beta_threshold=False,
            model_type=args.model_type,
            attack_config=attack_config,
            output_dir=os.path.join(args.output_dir, "manual_threshold"),
            log_level=args.log_level,
            enable_ensemble=args.enable_ensemble
        )
    else:
        # Run single experiment
        run_adaptive_trust_cifar10(
            num_actors=args.num_actors,
            num_rounds=args.num_rounds,
            topology_type=args.topology,
            trust_profile=args.trust_profile,
            use_beta_threshold=not args.disable_beta,
            model_type=args.model_type,
            attack_config=attack_config,
            output_dir=args.output_dir,
            log_level=args.log_level,
            enable_ensemble=args.enable_ensemble
        )


if __name__ == "__main__":
    main()