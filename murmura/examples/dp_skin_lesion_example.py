#!/usr/bin/env python3
"""
DP-Enhanced Skin Lesion Classification - Central Differential Privacy for Medical FL
Updated to include all arguments from the regular skin lesion example for consistency.
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as func

from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.model.pytorch_model import PyTorchModel, TorchModelWrapper
from murmura.privacy.dp_config import (
    DifferentialPrivacyConfig, DPMechanism, NoiseApplication,
    ClippingStrategy, DPAccountant
)
from murmura.privacy.dp_integration import (
    DPOrchestrationConfig, create_dp_federated_learning_process
)
from murmura.visualization.network_visualizer import NetworkVisualizer


# IDENTICAL WideResNet model components from normal skin lesion example
class BasicBlock(PyTorchModel):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.drop_rate = drop_rate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
                (not self.equalInOut)
                and nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
                or None
        )

    def forward(self, x):
        out = None
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = func.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(PyTorchModel):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self.make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate
        )

    @staticmethod
    def make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    drop_rate,
                    )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(PyTorchModel):
    """
    WideResNet model optimized for skin lesion classification.
    IDENTICAL to normal skin lesion example - no modifications.
    Default configuration targets HAM10000 dataset (7 classes).
    """

    def __init__(self, depth=16, num_classes=7, widen_factor=8, drop_rate=0.3):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.nChannels = n_channels[3]

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        # IDENTICAL forward pass from normal skin lesion example
        # Ensure input has correct format for medical images
        if x.dim() == 3:  # If missing batch dimension
            x = x.unsqueeze(0)

        # Medical images should be RGB (3 channels)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = func.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out


def create_skin_lesion_preprocessor(image_size: int = 128):
    """
    Create skin lesion specific data preprocessor.
    IDENTICAL function from normal skin lesion example.

    Args:
        image_size: Target image size for preprocessing

    Returns:
        Configured preprocessor for skin lesion images
    """
    try:
        from murmura.data_processing.data_preprocessor import create_image_preprocessor

        # Skin lesion specific configuration
        return create_image_preprocessor(
            grayscale=False,  # Medical images are typically RGB
            normalize=True,  # Normalize to [0,1]
            target_size=(image_size, image_size),  # Resize for consistent input
        )
    except ImportError:
        # Generic preprocessor not available, use automatic detection
        logging.getLogger("murmura.dp_skin_lesion").info(
            "Using automatic data type detection"
        )
        return None


def select_device(device_arg="auto"):
    """
    Select the appropriate device based on availability.
    IDENTICAL function from normal skin lesion example.

    Args:
        device_arg: Requested device ('auto', 'cuda', 'mps', 'cpu')

    Returns:
        device: Device to use
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("dp_skin_lesion_federated.log"),
        ],
    )


def create_dp_config(args) -> DifferentialPrivacyConfig:
    """Create differential privacy configuration for medical federated learning."""
    return DifferentialPrivacyConfig(
        # Core privacy parameters - stricter for medical data
        epsilon=args.epsilon,
        delta=args.delta,

        # Mechanism configuration - Central DP for better utility
        mechanism=DPMechanism.GAUSSIAN,
        noise_application=NoiseApplication.SERVER_SIDE,  # Central DP

        # Clipping configuration - adaptive for medical images
        clipping_strategy=ClippingStrategy.ADAPTIVE,
        clipping_norm=args.clipping_norm,
        target_quantile=0.5,

        # Privacy accounting
        accountant=DPAccountant.RDP,

        # Client sampling for privacy amplification
        client_sampling_rate=args.client_sampling_rate,

        # Per-client privacy for medical applications
        per_client_clipping=True,
        max_clients_per_user=1,  # Each client represents one hospital/institution

        # Monitoring
        enable_privacy_monitoring=True,
        privacy_budget_warning_threshold=0.8,

        # Total rounds for budget allocation
        total_rounds=args.rounds,
    )


def add_integer_labels_to_dataset(
        dataset: MDataset, logger: logging.Logger
) -> tuple[list[str], int, dict[str, int]]:
    """
    Add integer label column to dataset by converting string dx categories.
    IDENTICAL function from normal skin lesion example.
    Enhanced to store preprocessing metadata for multi-node compatibility.

    Args:
        dataset: The MDataset object to modify
        logger: Logger instance for logging

    Returns:
        tuple: (dx_categories, num_classes, dx_to_label_mapping)
    """
    # Get a sample from the training split to check label type
    train_split = dataset.get_split("train")
    sample_label = train_split["dx"][0]

    if isinstance(sample_label, str):
        logger.info("Converting string diagnostic categories to integer labels...")

        # Get all unique diagnostic categories across all splits
        all_dx_categories = set()

        for split_name in dataset.available_splits:
            split_data = dataset.get_split(split_name)
            split_dx = set(split_data["dx"])
            all_dx_categories.update(split_dx)

        # Sort for consistent ordering across all nodes/actors
        dx_categories = sorted(list(all_dx_categories))
        num_classes = len(dx_categories)

        # Create mapping from dx to integer label
        dx_to_label = {dx: i for i, dx in enumerate(dx_categories)}

        logger.info(f"Diagnostic categories ({num_classes}): {dx_categories}")
        logger.info(f"Label mapping: {dx_to_label}")

        # Define function to add label column
        def add_label_column(example):
            example["label"] = dx_to_label[example["dx"]]
            return example

        # Apply to all splits
        logger.info("Adding integer 'label' column to all dataset splits...")
        for split_name in dataset.available_splits:
            logger.debug(f"Processing split: {split_name}")
            split_data = dataset.get_split(split_name)

            # Apply the mapping
            split_data = split_data.map(add_label_column)

            # Update the dataset split
            dataset._splits[split_name] = split_data

            # Verify the mapping worked
            sample_new = split_data[0]
            logger.debug(
                f"Split {split_name}: dx='{sample_new['dx']}' -> label={sample_new['label']}"
            )

        logger.info("Successfully added integer 'label' column to all dataset splits")

        # CRITICAL: Store preprocessing information in dataset metadata for multi-node compatibility
        if (
                not hasattr(dataset, "_dataset_metadata")
                or dataset._dataset_metadata is None
        ):
            dataset._dataset_metadata = {}

        # Store the preprocessing information that remote nodes will need
        preprocessing_info = {
            "label_encoding": {
                "source_column": "dx",
                "target_column": "label",
                "mapping": dx_to_label,
            }
        }

        # Add to dataset metadata
        dataset._dataset_metadata["preprocessing_applied"] = preprocessing_info

        logger.info("Stored preprocessing metadata for multi-node compatibility")
        logger.debug(f"Preprocessing metadata: {preprocessing_info}")

        return dx_categories, num_classes, dx_to_label

    else:
        # Labels are already integers
        logger.info("Integer labels detected in dx column")

        # Get unique values to determine categories
        all_labels = set()
        for split_name in dataset.available_splits:
            split_data = dataset.get_split(split_name)
            split_labels = set(split_data["dx"])
            all_labels.update(split_labels)

        dx_categories = sorted(list(all_labels))
        num_classes = len(dx_categories)

        logger.info(
            f"Integer labels detected. Categories ({num_classes}): {dx_categories}"
        )

        # Add label column that's just a copy of dx
        def copy_dx_to_label(example):
            example["label"] = int(example["dx"])
            return example

        # Apply to all splits
        for split_name in dataset.available_splits:
            split_data = dataset.get_split(split_name)
            split_data = split_data.map(copy_dx_to_label)
            dataset._splits[split_name] = split_data

        # Create identity mapping
        dx_to_label = {str(dx): dx for dx in dx_categories}

        return [str(cat) for cat in dx_categories], num_classes, dx_to_label


def main():
    """Main function for DP-enhanced skin lesion classification."""
    parser = argparse.ArgumentParser(
        description="DP-Enhanced Federated Learning for Skin Cancer Classification (Central DP)"
    )

    # Core federated learning arguments (from regular skin lesion example)
    parser.add_argument(
        "--num_actors", type=int, default=5, help="Number of virtual clients (hospitals)"
    )
    parser.add_argument(
        "--partition_strategy",
        choices=["dirichlet", "iid"],
        default="dirichlet",
        help="Data partitioning strategy",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Dirichlet concentration parameter"
    )
    parser.add_argument(
        "--min_partition_size",
        type=int,
        default=50,
        help="Minimum samples per partition",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--test_split", type=str, default="test", help="Test split to use"
    )
    parser.add_argument(
        "--validation_split",
        type=str,
        default="validation",
        help="Validation split to use",
    )
    parser.add_argument(
        "--aggregation_strategy",
        type=str,
        choices=["fedavg", "trimmed_mean"],
        default="fedavg",
        help="Aggregation strategy to use",
    )
    parser.add_argument(
        "--trim_ratio",
        type=float,
        default=0.1,
        help="Trim ratio for trimmed_mean strategy",
    )

    # Topology arguments (from regular skin lesion example)
    parser.add_argument(
        "--topology",
        type=str,
        default="star",
        choices=["star", "ring", "complete", "line", "custom"],
        help="Network topology between clients",
    )
    parser.add_argument(
        "--hub_index", type=int, default=0, help="Hub node index for star topology"
    )

    # Training arguments (from regular skin lesion example)
    parser.add_argument(
        "--rounds", type=int, default=5, help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of local epochs per round"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate in WideResNet"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="dp_skin_cancer_federated_model.pt",
        help="Path to save the final model",
    )

    # Skin lesion specific arguments (from regular skin lesion example)
    parser.add_argument(
        "--image_size", type=int, default=128, help="Size to resize images to"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="marmal88/skin_cancer",
        help="Skin lesion dataset name",
    )
    parser.add_argument(
        "--widen_factor", type=int, default=8, help="WideResNet widen factor"
    )
    parser.add_argument("--depth", type=int, default=16, help="WideResNet depth")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--debug_data",
        action="store_true",
        help="Print debug information about skin lesion data format",
    )

    # Multi-node Ray cluster arguments (from regular skin lesion example)
    parser.add_argument(
        "--ray_address",
        type=str,
        default=None,
        help="Ray cluster address. If None, uses local cluster.",
    )
    parser.add_argument(
        "--ray_namespace",
        type=str,
        default="murmura_dp_skin_lesion",
        help="Ray namespace for isolation",
    )
    parser.add_argument(
        "--actors_per_node",
        type=int,
        default=None,
        help="Number of actors per physical node. If None, distributes evenly.",
    )
    parser.add_argument(
        "--cpus_per_actor",
        type=float,
        default=1.0,
        help="CPU resources per actor",
    )
    parser.add_argument(
        "--gpus_per_actor",
        type=float,
        default=None,
        help="GPU resources per actor. If None, auto-calculated.",
    )
    parser.add_argument(
        "--memory_per_actor",
        type=int,
        default=None,
        help="Memory (MB) per actor",
    )
    parser.add_argument(
        "--placement_strategy",
        type=str,
        choices=["spread", "pack", "strict_spread", "strict_pack"],
        default="spread",
        help="Actor placement strategy across nodes",
    )
    parser.add_argument(
        "--auto_detect_cluster",
        action="store_true",
        help="Auto-detect Ray cluster from environment variables",
    )

    # Logging and monitoring arguments (from regular skin lesion example)
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--monitor_resources",
        action="store_true",
        help="Monitor and log resource usage during training",
    )

    # Visualization arguments (from regular skin lesion example)
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--create_animation",
        action="store_true",
        help="Create animation of the training process",
    )
    parser.add_argument(
        "--create_frames",
        action="store_true",
        help="Create individual frames of the training process",
    )
    parser.add_argument(
        "--create_summary",
        action="store_true",
        help="Create summary plot of the training process",
    )

    # DP-specific arguments - stricter for medical data
    parser.add_argument(
        "--epsilon", type=float, default=0.5,
        help="Privacy budget (ε) - lower for medical data (stronger privacy)"
    )
    parser.add_argument(
        "--delta", type=float, default=1e-6,
        help="Failure probability (δ) - lower for medical data"
    )
    parser.add_argument(
        "--clipping_norm", type=float, default=1.0,
        help="Gradient clipping threshold"
    )
    parser.add_argument(
        "--client_sampling_rate", type=float, default=0.8,
        help="Client sampling rate for privacy amplification"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger("murmura.dp_skin_lesion")

    try:
        # Determine the device to use - IDENTICAL to normal skin lesion example
        device = select_device(args.device)
        logger.info(f"Using {device.upper()} device for training")

        # Create DP configuration for medical federated learning
        dp_config = create_dp_config(args)
        logger.info("Created DP configuration for medical federated learning")

        # Create enhanced configuration with multi-node support
        from murmura.aggregation.aggregation_config import AggregationConfig, AggregationStrategyType
        from murmura.network_management.topology import TopologyConfig, TopologyType
        from murmura.node.resource_config import RayClusterConfig, ResourceConfig

        ray_cluster_config = RayClusterConfig(
            address=args.ray_address,
            namespace=args.ray_namespace,
            logging_level=args.log_level,
            auto_detect_cluster=args.auto_detect_cluster,
        )

        resource_config = ResourceConfig(
            actors_per_node=args.actors_per_node,
            cpus_per_actor=args.cpus_per_actor,
            gpus_per_actor=args.gpus_per_actor,
            memory_per_actor=args.memory_per_actor,
            placement_strategy=args.placement_strategy,
        )

        config = DPOrchestrationConfig(
            # Core FL parameters
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            min_partition_size=args.min_partition_size,
            split=args.split,

            # Network topology
            topology=TopologyConfig(
                topology_type=TopologyType(args.topology),
                hub_index=args.hub_index
            ),

            # Aggregation strategy
            aggregation=AggregationConfig(
                strategy_type=AggregationStrategyType(args.aggregation_strategy),
                params={"trim_ratio": args.trim_ratio}
                if args.aggregation_strategy == "trimmed_mean"
                else None,
            ),

            # Dataset configuration
            dataset_name=args.dataset_name,
            ray_cluster=ray_cluster_config,
            resources=resource_config,

            # CRITICAL: Must specify feature and label columns
            feature_columns=["image"],  # Skin lesion images
            label_column="label",       # Diagnostic labels

            # Differential Privacy configuration (Central DP)
            differential_privacy=dp_config,
            enable_privacy_dashboard=True
        )

        logger.info("Created DP orchestration configuration")

        logger.info("=== Loading Skin Lesion Dataset ===")
        # IDENTICAL dataset loading from normal skin lesion example
        train_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name=args.dataset_name,
            split=args.split,
        )

        test_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name=args.dataset_name,
            split=args.test_split,
        )

        validation_dataset = MDataset.load_dataset_with_multinode_support(
            DatasetSource.HUGGING_FACE,
            dataset_name=args.dataset_name,
            split=args.validation_split,
        )

        # Merge datasets to have all splits available
        train_dataset.merge_splits(test_dataset)
        train_dataset.merge_splits(validation_dataset)
        logger.info("Loaded and merged skin lesion dataset")

        # Convert string labels to integers and get diagnostic categories
        # IDENTICAL to normal skin lesion example
        logger.info("=== Processing Labels ===")
        dx_categories, num_classes, dx_to_label = add_integer_labels_to_dataset(
            train_dataset, logger
        )

        # Debug skin lesion data format if requested
        if args.debug_data:
            logger.info("=== Debugging Skin Lesion Data Format ===")
            try:
                split_dataset = train_dataset.get_split(args.split)
                feature_data = split_dataset["image"]

                logger.info(f"Dataset: {args.dataset_name}")
                logger.info(f"Split: {args.split}")
                logger.info(f"Number of samples: {len(feature_data)}")
                logger.info(f"Diagnostic categories: {dx_categories}")
                logger.info(f"Label mapping: {dx_to_label}")

                if len(feature_data) > 0:
                    sample = feature_data[0]
                    logger.info(f"Sample type: {type(sample)}")
                    logger.info(f"Sample shape: {getattr(sample, 'shape', 'N/A')}")
                    logger.info(f"Sample mode: {getattr(sample, 'mode', 'N/A')}")
                    if hasattr(sample, "size"):
                        logger.info(f"Sample size: {sample.size}")

                    # Check label conversion
                    sample_full = split_dataset[0]
                    logger.info(
                        f"Sample dx: '{sample_full['dx']}' -> label: {sample_full['label']}"
                    )

            except Exception as e:
                logger.error(f"Error debugging skin lesion data format: {e}")

        logger.info("=== Creating WideResNet Model ===")
        # Create the WideResNet model for skin lesion classification
        # IDENTICAL to normal skin lesion example
        model = WideResNet(
            depth=args.depth,
            num_classes=num_classes,
            widen_factor=args.widen_factor,
            drop_rate=args.dropout,
        )

        logger.info(
            f"Created WideResNet: depth={args.depth}, widen_factor={args.widen_factor}, "
            f"num_classes={num_classes}, dropout={args.dropout}"
        )

        # Create skin lesion specific data preprocessor
        # IDENTICAL to normal skin lesion example
        skin_lesion_preprocessor = create_skin_lesion_preprocessor(args.image_size)

        # Create model wrapper with skin lesion specific configuration
        # IDENTICAL to normal skin lesion example
        input_shape = (3, args.image_size, args.image_size)  # RGB images
        global_model = TorchModelWrapper(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
            input_shape=input_shape,
            device=device,
            data_preprocessor=skin_lesion_preprocessor,
        )
        logger.info("Created model wrapper")

        logger.info("=== Setting Up DP-Enhanced Learning Process ===")
        learning_process = create_dp_federated_learning_process(
            config, train_dataset, global_model
        )
        logger.info("Created DP-enhanced federated learning process")

        # Set up visualization if requested
        visualizer = None
        if args.create_animation or args.create_frames or args.create_summary:
            logger.info("=== Setting Up Visualization ===")
            vis_dir = os.path.join(
                args.vis_dir, f"dp_skin_cancer_{args.topology}_{args.aggregation_strategy}"
            )
            os.makedirs(vis_dir, exist_ok=True)

            visualizer = NetworkVisualizer(output_dir=vis_dir)
            learning_process.register_observer(visualizer)
            logger.info("Registered visualizer with learning process")

        try:
            # Initialize the learning process
            partitioner = PartitionerFactory.create(config)
            logger.info("Created partitioner")

            logger.info("=== Initializing Learning Process ===")
            learning_process.initialize(
                num_actors=config.num_actors,
                topology_config=config.topology,
                aggregation_config=config.aggregation,
                partitioner=partitioner
            )

            # Monitor dataset distribution status
            dist_status = learning_process.get_dataset_distribution_status()
            logger.info(f"Dataset distribution: {dist_status['distribution_strategy']}")
            logger.info(
                f"Hospitals ready: {dist_status['healthy_actors']}/{dist_status['total_actors']}"
            )
            if dist_status["distribution_strategy"] == "lazy":
                logger.info(
                    f"Lazy loading enabled on {dist_status['lazy_loading_actors']} hospitals"
                )

            # Monitor memory usage
            if args.monitor_resources:
                memory_status = learning_process.monitor_memory_usage()
                logger.info(
                    f"Cluster memory: {memory_status.get('available_memory_gb', 'N/A')}GB available"
                )
                if memory_status.get("high_usage_nodes", 0) > 0:
                    logger.warning(
                        f"{memory_status['high_usage_nodes']} nodes have high memory usage"
                    )

            # Get and log cluster information
            cluster_summary = learning_process.get_cluster_summary()
            logger.info("=== DP-Enhanced Cluster Summary ===")
            logger.info(
                f"Cluster type: {cluster_summary.get('cluster_type', 'unknown')}"
            )
            logger.info(f"Total nodes: {cluster_summary.get('total_nodes', 'unknown')}")
            logger.info(
                f"Total hospitals: {cluster_summary.get('total_actors', 'unknown')}"
            )

            # Log privacy configuration
            privacy_desc = dp_config.get_privacy_description()
            utility_impact = dp_config.estimate_utility_impact()

            logger.info("=== Privacy Configuration for Medical Data ===")
            logger.info(f"Privacy Guarantee: {privacy_desc}")
            logger.info(f"Privacy Level: {utility_impact['privacy_level']}")
            logger.info(f"Expected Utility Impact: {utility_impact['utility_impact']}")
            logger.info(f"Communication Impact: {utility_impact['communication']}")
            logger.info(f"DP Mode: Central (server-side noise)")
            logger.info(f"Per-client Privacy: {dp_config.per_client_clipping}")

            # Print initial summary
            logger.info("=== DP-Enhanced Skin Lesion Federated Learning Setup ===")
            logger.info(f"Dataset: {args.dataset_name}")
            logger.info(f"Partitioning: {config.partition_strategy} (α={args.alpha})")
            logger.info(f"Hospitals: {config.num_actors}")
            logger.info(f"Aggregation: {config.aggregation.strategy_type}")
            logger.info(f"Topology: {config.topology.topology_type}")
            logger.info(f"Privacy: Central DP (ε={args.epsilon}, δ={args.delta})")
            logger.info(f"Clipping norm: {args.clipping_norm}")
            logger.info(f"Client sampling rate: {args.client_sampling_rate}")
            logger.info(f"Rounds: {args.rounds}")
            logger.info(f"Local epochs: {args.epochs}")
            logger.info(f"Batch size: {args.batch_size}")
            logger.info(f"Learning rate: {args.lr}")
            logger.info(f"Weight decay: {args.weight_decay}")
            logger.info(f"Device: {device}")
            logger.info(f"Image size: {args.image_size}")
            logger.info(f"Classes ({num_classes}): {dx_categories}")
            logger.info(f"Using feature column: {config.feature_columns}")
            logger.info(f"Using label column: {config.label_column}")

            logger.info("=== Starting DP-Enhanced Medical FL Training ===")

            # Execute training
            results = learning_process.execute()

            # Generate visualizations if requested
            if visualizer and (
                    args.create_animation or args.create_frames or args.create_summary
            ):
                logger.info("=== Generating Visualizations ===")

                if args.create_animation:
                    logger.info("Creating animation...")
                    visualizer.render_training_animation(
                        filename=f"dp_skin_cancer_{args.topology}_{args.aggregation_strategy}_animation.mp4",
                        fps=2,
                    )

                if args.create_frames:
                    logger.info("Creating frame sequence...")
                    visualizer.render_frame_sequence(
                        prefix=f"dp_skin_cancer_{args.topology}_{args.aggregation_strategy}_step"
                    )

                if args.create_summary:
                    logger.info("Creating summary plot...")
                    visualizer.render_summary_plot(
                        filename=f"dp_skin_cancer_{args.topology}_{args.aggregation_strategy}_summary.png"
                    )

            # Save model with comprehensive metadata - IDENTICAL to normal skin lesion example
            logger.info("=== Saving Model ===")
            save_path = args.save_path

            # Create comprehensive checkpoint for medical model
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": global_model.optimizer.state_dict(),
                "dx_categories": dx_categories,
                "dx_to_label_mapping": dx_to_label,
                "num_classes": num_classes,
                "depth": args.depth,
                "widen_factor": args.widen_factor,
                "dropout": args.dropout,
                "image_size": args.image_size,
                "input_shape": input_shape,
                "device": device,
                "learning_type": "centralized_dp",  # Mark as DP-enhanced
                "dp_config": {
                    "epsilon": dp_config.epsilon,
                    "delta": dp_config.delta,
                    "mechanism": dp_config.mechanism.value,
                    "noise_application": dp_config.noise_application.value,
                    "clipping_norm": dp_config.clipping_norm,
                },
                "config": {
                    k: v for k, v in vars(args).items() if not k.startswith("_")
                },
            }

            os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
            torch.save(checkpoint, save_path)
            logger.info(f"DP-enhanced skin lesion model saved to '{save_path}'")

            # Print final summary
            logger.info("=== DP-Enhanced Medical FL Training Results ===")
            logger.info(f"Initial Accuracy: {results['initial_metrics']['accuracy']:.4f}")
            logger.info(f"Final Accuracy: {results['final_metrics']['accuracy']:.4f}")
            logger.info(f"Accuracy Improvement: {results['accuracy_improvement']:.4f}")
            logger.info(f"Training device: {device}")
            logger.info(f"Diagnostic Categories: {dx_categories}")

            if 'privacy_summary' in results:
                privacy_summary = results['privacy_summary']
                if privacy_summary['dp_enabled']:
                    logger.info("Differential Privacy: ENABLED (Central DP)")
                    if 'accountant' in privacy_summary:
                        spent = privacy_summary['accountant']['spent']
                        total = privacy_summary['accountant']['total_budget']
                        logger.info(f"Privacy Spent: ε={spent['epsilon']:.4f}/{total['epsilon']:.4f}, "
                                    f"δ={spent['delta']:.2e}/{total['delta']:.2e}")

                        # Calculate privacy budget utilization
                        eps_utilization = (spent['epsilon'] / total['epsilon']) * 100
                        delta_utilization = (spent['delta'] / total['delta']) * 100
                        logger.info(f"Privacy Budget Utilization: ε={eps_utilization:.1f}%, δ={delta_utilization:.1f}%")
                else:
                    logger.info("Differential Privacy: DISABLED")

            # Display detailed metrics if available
            if "round_metrics" in results:
                logger.info("=== Detailed Round Metrics ===")
                for round_data in results["round_metrics"]:
                    round_num = round_data["round"]
                    logger.info(f"Round {round_num}:")
                    logger.info(
                        f"  Train Loss: {round_data.get('train_loss', 'N/A'):.4f}"
                    )
                    logger.info(
                        f"  Train Accuracy: {round_data.get('train_accuracy', 'N/A'):.4f}"
                    )
                    logger.info(
                        f"  Test Loss: {round_data.get('test_loss', 'N/A'):.4f}"
                    )
                    logger.info(
                        f"  Test Accuracy: {round_data.get('test_accuracy', 'N/A'):.4f}"
                    )

            # Medical-specific logging
            logger.info("=== Medical FL Compliance Summary ===")
            logger.info(f"Privacy-preserving: YES (ε={dp_config.epsilon}, δ={dp_config.delta})")
            logger.info(f"Per-hospital privacy: {dp_config.per_client_clipping}")
            logger.info(f"Data never leaves hospitals: YES (federated learning)")
            logger.info(f"Noise added to prevent reconstruction: YES ({dp_config.mechanism.value})")

        finally:
            logger.info("=== Shutting Down ===")
            learning_process.shutdown()

    except Exception as e:
        logger.error(f"DP-enhanced medical federated learning failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Clean up
        if 'learning_process' in locals():
            learning_process.shutdown()
        logger.info("Training completed and resources cleaned up")


if __name__ == "__main__":
    main()
