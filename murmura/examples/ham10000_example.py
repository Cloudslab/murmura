import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.data_processing.dataset import MDataset
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.model.pytorch_model import PyTorchModel, TorchModelWrapper
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.orchestration.learning_process.federated_learning_process import (
    FederatedLearningProcess,
)
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.visualization.network_visualizer import NetworkVisualizer


# Define the WideResNet model
class BasicBlock(PyTorchModel):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
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
        self.droprate = dropRate
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
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(PyTorchModel):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(PyTorchModel):
    """
    WideResNet model with adjustable depth, width, and number of classes.
    Default configuration is optimized for skin lesion classification (7 classes for HAM10000).
    """

    def __init__(self, depth=16, num_classes=7, widen_factor=8, dropRate=0.3):
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
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, dropRate)
        # Global average pooling and classifier
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
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(
            out, 1
        )  # Use adaptive pooling for variable input sizes
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out


class CustomDatasetWrapper:
    """Custom utility to preprocess and adapt the HAM10000 dataset"""

    @staticmethod
    def prepare_dataset(dataset_name="marmal88/skin_cancer", image_size=224):
        # Load the HAM10000 dataset
        print(f"Loading dataset: {dataset_name}")
        raw_dataset = load_dataset(dataset_name)

        # Extract diagnostic categories and create a label mapping
        dx_categories = sorted(set(raw_dataset["train"]["dx"]))
        dx_to_label = {dx: i for i, dx in enumerate(dx_categories)}

        print(f"Diagnostic categories: {dx_categories}")
        print(f"Label mapping: {dx_to_label}")

        # Define image transformation
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Define preprocessing function to convert dx categories to numeric labels
        # and preprocess images to standard size and format
        def preprocess_example(example):
            # Add a numeric label based on the dx category
            example["label"] = dx_to_label[example["dx"]]

            # Preprocess the image - transform to tensor and convert to PyTorch format
            pil_image = example["image"]

            # Apply the transformation and convert to numpy array
            # This gives us a tensor in CHW format of shape [3, image_size, image_size]
            transformed_tensor = transform(pil_image)

            # Store preprocessed tensor as 'image_tensor' - convert to numpy for storage
            example["image_tensor"] = transformed_tensor.numpy()

            return example

        # Apply preprocessing to all splits
        print("\nPreprocessing images and adding labels...")
        processed_dataset = {}
        for split in raw_dataset.keys():
            print(f"Processing {split} split...")
            processed_dataset[split] = raw_dataset[split].map(
                preprocess_example, desc=f"Processing {split}"
            )

        # Convert to DatasetDict for compatibility
        processed_dataset = DatasetDict(processed_dataset)

        # Print dataset statistics
        print("\nDataset statistics:")
        for split in processed_dataset.keys():
            print(f"  {split}: {len(processed_dataset[split])} examples")

        # Label distribution
        for split in processed_dataset.keys():
            label_counts = {}
            for label in processed_dataset[split]["label"]:
                label_counts[label] = label_counts.get(label, 0) + 1

            print(f"\nLabel distribution in {split} split:")
            for label, count in sorted(label_counts.items()):
                dx = dx_categories[label]
                percentage = 100 * count / len(processed_dataset[split])
                print(f"  {dx} (label {label}): {count} examples ({percentage:.2f}%)")

        return processed_dataset, dx_categories


def select_device(device_arg="auto"):
    """
    Select the appropriate device based on availability

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


def main() -> None:
    """
    Orchestrate Learning Process for skin cancer classification
    """
    parser = argparse.ArgumentParser(
        description="Federated Learning for Skin Cancer Classification"
    )
    parser.add_argument(
        "--num_actors", type=int, default=5, help="Number of virtual clients"
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
        help="Trim ratio for trimmed_mean strategy (0.1 = 10% trimmed from each end)",
    )

    # Topology arguments
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

    # Training arguments
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
        default="skin_cancer_federated_model.pt",
        help="Path to save the final model",
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Size to resize images to"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="marmal88/skin_cancer", help="Dataset name"
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
        help="Device to use for training (auto selects best available)",
    )

    # Visualization arguments
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

    args = parser.parse_args()

    try:
        # Determine the device to use
        device = select_device(args.device)
        print(f"\n=== Using {device.upper()} device for training ===")

        # Process the raw dataset to add a 'label' column mapping from 'dx'
        print("\n=== Preparing HAM10000 Skin Cancer Dataset ===")
        processed_dataset, dx_categories = CustomDatasetWrapper.prepare_dataset(
            dataset_name=args.dataset_name, image_size=args.image_size
        )

        # Now that we've processed the dataset and added a 'label' column,
        # we can define the number of classes
        num_classes = len(dx_categories)
        print(f"Number of classes: {num_classes}")

        # Create configuration from command-line arguments
        config = OrchestrationConfig(
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            min_partition_size=args.min_partition_size,
            split=args.split,
            topology=TopologyConfig(
                topology_type=TopologyType(args.topology), hub_index=args.hub_index
            ),
            aggregation=AggregationConfig(
                strategy_type=AggregationStrategyType(args.aggregation_strategy),
                params={"trim_ratio": args.trim_ratio}
                if args.aggregation_strategy == "trimmed_mean"
                else None,
            ),
            dataset_name=args.dataset_name,  # Set the dataset name
        )

        # Add additional configuration needed for the learning process
        process_config = config.model_dump()
        process_config.update(
            {
                "rounds": args.rounds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "test_split": args.test_split,
                "validation_split": args.validation_split,
                # Use the preprocessed image_tensor instead of raw image
                "feature_columns": ["image_tensor"],
                "label_column": "label",  # Use the 'label' field we added in the preprocessing
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "image_size": args.image_size,
                "dx_categories": dx_categories,
                "widen_factor": args.widen_factor,
                "depth": args.depth,
                "dropout": args.dropout,
            }
        )

        print("\n=== Converting Dataset to Murmura Format ===")

        # We need to convert our processed HuggingFace dataset into Murmura's MDataset format
        # Create MDatasets from our processed DatasetDict
        train_split = processed_dataset[config.split]
        test_split = processed_dataset[args.test_split]
        validation_split = processed_dataset[args.validation_split]

        # Convert to Murmura's format
        # Create Murmura datasets
        train_dataset = MDataset(DatasetDict({config.split: train_split}))
        test_dataset = MDataset(DatasetDict({args.test_split: test_split}))
        validation_dataset = MDataset(
            DatasetDict({args.validation_split: validation_split})
        )

        # Merge datasets to have all splits available
        train_dataset.merge_splits(test_dataset)
        train_dataset.merge_splits(validation_dataset)

        print("\n=== Creating Data Partitions ===")
        # Create partitioner
        partitioner = PartitionerFactory.create(config)

        print("\n=== Creating and Initializing Model ===")
        # Create the WideResNet model with PyTorch wrapper
        model = WideResNet(
            depth=args.depth,
            num_classes=num_classes,
            widen_factor=args.widen_factor,
            dropRate=args.dropout,
        )

        print(
            f"Created WideResNet with depth={args.depth}, widen_factor={args.widen_factor}, "
            f"num_classes={num_classes}, dropout={args.dropout}"
        )

        # Create a custom TorchModelWrapper extension that handles our preprocessed tensors properly
        class PreprocessedTensorModelWrapper(TorchModelWrapper):
            def _prepare_data(self, data, labels=None, batch_size=32):
                """
                Override the base _prepare_data method to handle our preprocessed tensors properly

                data: Numpy array containing our preprocessed tensors (already in CHW format)
                """
                tensor_data = torch.tensor(data, dtype=torch.float32)

                # Our data is already in the right shape, so no reshaping needed
                # The input data should already be [N, C, H, W] where C=3, H=W=224

                if labels is not None:
                    tensor_labels = torch.tensor(labels, dtype=torch.long)
                    dataset = torch.utils.data.TensorDataset(tensor_data, tensor_labels)
                else:
                    dataset = torch.utils.data.TensorDataset(tensor_data)

                return torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=(labels is not None)
                )

        # Create the model wrapper
        # Input shape is [C, H, W] - note that it's already in CHW format from our preprocessing
        input_shape = (3, args.image_size, args.image_size)

        global_model = PreprocessedTensorModelWrapper(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
            input_shape=input_shape,
            device=device,
        )

        print("\n=== Setting Up Learning Process ===")
        # Create standard learning process with our custom preprocessing
        learning_process = FederatedLearningProcess(
            config=process_config,
            dataset=train_dataset,
            model=global_model,
        )

        # Set up visualization if requested
        visualizer = None
        if args.create_animation or args.create_frames or args.create_summary:
            print("\n=== Setting Up Visualization ===")
            # Create visualization directory
            vis_dir = os.path.join(
                args.vis_dir, f"skin_cancer_{args.topology}_{args.aggregation_strategy}"
            )
            os.makedirs(vis_dir, exist_ok=True)

            # Create visualizer
            visualizer = NetworkVisualizer(output_dir=vis_dir)

            # Register visualizer with learning process
            learning_process.register_observer(visualizer)
            print("Registered visualizer with learning process")
            print(
                f"Current observers: {len(learning_process.training_monitor.observers)}"
            )

        try:
            # Initialize the learning process
            print("\n=== Initializing Learning Process ===")
            learning_process.initialize(
                num_actors=config.num_actors,
                topology_config=config.topology,
                aggregation_config=config.aggregation,
                partitioner=partitioner,
            )

            # Print initial summary
            print("\n=== Federated Learning Setup ===")
            print(f"Dataset: {args.dataset_name}")
            print(f"Partition strategy: {config.partition_strategy}")
            print(f"Clients: {config.num_actors}")
            print(f"Aggregation strategy: {config.aggregation.strategy_type}")
            print(f"Topology: {config.topology.topology_type}")
            print(f"Rounds: {args.rounds}")
            print(f"Local epochs: {args.epochs}")
            print(f"Batch size: {args.batch_size}")
            print(f"Learning rate: {args.lr}")
            print(f"Weight decay: {args.weight_decay}")
            print(f"Device: {device}")
            print(f"Classes ({num_classes}): {dx_categories}")

            print("\n=== Starting Federated Learning ===")
            # Execute the learning process
            results = learning_process.execute()

            # Generate visualizations if requested
            if visualizer and (
                args.create_animation or args.create_frames or args.create_summary
            ):
                print("\n=== Generating Visualizations ===")

                if args.create_animation:
                    print("Creating animation...")
                    visualizer.render_training_animation(
                        filename=f"skin_cancer_{args.topology}_{args.aggregation_strategy}_animation.mp4",
                        fps=2,
                    )

                if args.create_frames:
                    print("Creating frame sequence...")
                    visualizer.render_frame_sequence(
                        prefix=f"skin_cancer_{args.topology}_{args.aggregation_strategy}_step"
                    )

                if args.create_summary:
                    print("Creating summary plot...")
                    visualizer.render_summary_plot(
                        filename=f"skin_cancer_{args.topology}_{args.aggregation_strategy}_summary.png"
                    )

            # Save the final model along with the class labels
            print("\n=== Saving Final Model ===")
            save_path = args.save_path

            # Save model with additional metadata
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": global_model.optimizer.state_dict(),
                "dx_categories": dx_categories,
                "num_classes": num_classes,
                "depth": args.depth,
                "widen_factor": args.widen_factor,
                "dropout": args.dropout,
                "image_size": args.image_size,
                "input_shape": input_shape,
                "device": device,
                "config": {
                    k: v for k, v in vars(args).items() if not k.startswith("_")
                },
            }

            os.makedirs(
                os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True
            )
            torch.save(checkpoint, save_path)
            print(f"Model saved to '{save_path}'")

            # Print final results
            print("\n=== Training Results ===")
            print(f"Initial accuracy: {results['initial_metrics']['accuracy']:.4f}")
            print(f"Final accuracy: {results['final_metrics']['accuracy']:.4f}")
            print(f"Accuracy improvement: {results['accuracy_improvement']:.4f}")
            print(f"Training device: {device}")

            # Display detailed metrics if available
            if "round_metrics" in results:
                print("\n--- Detailed Round Metrics ---")
                for round_data in results["round_metrics"]:
                    round_num = round_data["round"]
                    print(f"Round {round_num}:")
                    print(f"  Train Loss: {round_data.get('train_loss', 'N/A'):.4f}")
                    print(
                        f"  Train Accuracy: {round_data.get('train_accuracy', 'N/A'):.4f}"
                    )
                    print(f"  Test Loss: {round_data.get('test_loss', 'N/A'):.4f}")
                    print(
                        f"  Test Accuracy: {round_data.get('test_accuracy', 'N/A'):.4f}"
                    )

        finally:
            print("\n=== Shutting Down ===")
            learning_process.shutdown()

    except Exception as e:
        print(f"Learning Process orchestration failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
