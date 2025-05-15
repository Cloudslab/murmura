import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm


# Define the same WideResNet model architecture for consistency
class BasicBlock(nn.Module):
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


class NetworkBlock(nn.Module):
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


class WideResNet(nn.Module):
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
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.nChannels = n_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        # Ensure input has the right shape for the network
        if x.dim() == 3:  # If missing batch dimension
            x = x.unsqueeze(0)

        # If grayscale, convert to RGB
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

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


def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a trained WideResNet model from checkpoint.

    Args:
        model_path (str): Path to the saved model checkpoint
        device (str): Device to load the model on ('cuda' or 'cpu')

    Returns:
        model (nn.Module): The loaded model
        dx_categories (list): List of diagnostic categories
        config (dict): Model configuration
    """
    print(f"Loading model from {model_path}")
    # Load the saved model checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    # Extract model configuration
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Create default configuration if not available
        config = {
            "depth": checkpoint.get("depth", 16),
            "widen_factor": checkpoint.get("widen_factor", 8),
            "dropout": checkpoint.get("dropout", 0.3),
            "num_classes": checkpoint.get("num_classes", 7),
        }

    # Extract class labels (diagnostic categories)
    dx_categories = checkpoint.get("dx_categories", None)
    num_classes = checkpoint.get("num_classes", 7)

    # Print model configuration
    print(f"Model configuration:")
    print(f"  Depth: {config.get('depth', 16)}")
    print(f"  Widen factor: {config.get('widen_factor', 8)}")
    print(f"  Dropout: {config.get('dropout', 0.3)}")
    print(f"  Number of classes: {num_classes}")

    if dx_categories:
        print(f"  Diagnostic categories: {dx_categories}")

    # Initialize model
    model = WideResNet(
        depth=config.get("depth", 16),
        num_classes=num_classes,
        widen_factor=config.get("widen_factor", 8),
        dropRate=config.get("dropout", 0.3),
    )

    # Load model state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, dx_categories, config


def get_transforms(image_size=224):
    """
    Create image transform pipeline.

    Args:
        image_size (int): Size to resize images to

    Returns:
        transforms: Transformation pipeline for preprocessing images
    """
    # Define image transforms for evaluation
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform


def predict_image(
    image_path,
    model,
    transform,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dx_categories=None,
):
    """
    Make prediction for a single image.

    Args:
        image_path (str): Path to image file
        model (nn.Module): Trained model
        transform: Image transform pipeline
        device (str): Device to use for inference
        dx_categories (list): List of diagnostic categories

    Returns:
        dict: Dictionary containing prediction results
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    # Get class name if available
    if dx_categories and predicted_class < len(dx_categories):
        class_label = dx_categories[predicted_class]
    else:
        class_label = f"Class {predicted_class}"

    # Return prediction results
    return {
        "predicted_class": predicted_class,
        "class_label": class_label,
        "confidence": confidence,
        "probabilities": probs[0].cpu().numpy(),
    }


def evaluate_model(
    model,
    test_dataset,
    transform,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dx_categories=None,
    batch_size=32,
    preprocess_tensor_available=False,
):
    """
    Evaluate model on a test dataset.

    Args:
        model (nn.Module): Trained model
        test_dataset: Test dataset
        transform: Image transform pipeline
        device (str): Device to use for evaluation
        dx_categories (list): List of diagnostic categories
        batch_size (int): Batch size for evaluation
        preprocess_tensor_available (bool): Whether the dataset has preprocessed tensors

    Returns:
        dict: Dictionary containing evaluation results
    """
    model.eval()

    print("Preparing test dataset...")
    # Get the number of samples in the test dataset
    num_samples = len(test_dataset)

    # Create dataloaders
    all_labels = []
    all_preds = []
    all_probs = []

    print(f"Evaluating {num_samples} test samples...")
    # Process test dataset in batches
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_indices = range(i, min(i + batch_size, num_samples))
            batch_images = []
            batch_labels = []

            # Prepare a batch of samples
            for idx in batch_indices:
                sample = test_dataset[idx]
                label = sample["label"]
                batch_labels.append(label)

                # Check if preprocessed tensor is available
                if preprocess_tensor_available and "image_tensor" in sample:
                    # Use the preprocessed tensor
                    image_tensor = torch.tensor(
                        sample["image_tensor"], dtype=torch.float32
                    )
                    batch_images.append(image_tensor)
                else:
                    # Use the original image and apply transformation
                    image = sample["image"]
                    image_tensor = transform(image)
                    batch_images.append(image_tensor)

            # Stack all images in the batch
            batch_images = torch.stack(batch_images).to(device)

            # Make predictions for the batch
            outputs = model(batch_images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Save results
            all_labels.extend(batch_labels)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    print(f"Evaluation results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Generate classification report
    report = classification_report(
        all_labels, all_preds, target_names=dx_categories, output_dict=True
    )

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Return evaluation results
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": all_labels,
        "predictions": all_preds,
        "probabilities": all_probs,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def preprocess_dataset(dataset, transform, dx_categories=None):
    """
    Preprocess the entire dataset to add image_tensor field.
    This is for compatibility with models trained using our modified training script.

    Args:
        dataset: Dataset to preprocess
        transform: Transform pipeline to apply
        dx_categories: List of diagnostic categories

    Returns:
        processed_dataset: Dataset with added image_tensor field
    """
    print("Preprocessing dataset to add image_tensor field...")

    # Define preprocessing function
    def preprocess_example(example):
        # Convert dx to label if not already present
        if "label" not in example and "dx" in example and dx_categories:
            dx_to_label = {dx: i for i, dx in enumerate(dx_categories)}
            example["label"] = dx_to_label[example["dx"]]

        # Apply transform to get tensor
        pil_image = example["image"]
        transformed_tensor = transform(pil_image)

        # Store the tensor in CHW format
        example["image_tensor"] = transformed_tensor.numpy()

        return example

    # Apply preprocessing to dataset
    processed_dataset = dataset.map(preprocess_example, desc="Preprocessing images")

    return processed_dataset


def visualize_results(results, dx_categories, output_dir="./results"):
    """
    Visualize evaluation results.

    Args:
        results (dict): Evaluation results
        dx_categories (list): List of diagnostic categories
        output_dir (str): Directory to save visualization results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    cm = results["confusion_matrix"]
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=dx_categories,
        yticklabels=dx_categories,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)

    # Create precision, recall, f1 plot for each class
    plt.figure(figsize=(12, 6))

    # Extract metrics for each class
    metrics = []
    for i, class_name in enumerate(dx_categories):
        if str(i) in results["classification_report"]:
            class_report = results["classification_report"][str(i)]
        else:
            class_report = results["classification_report"][class_name]

        metrics.append(
            {
                "class": class_name,
                "precision": class_report["precision"],
                "recall": class_report["recall"],
                "f1-score": class_report["f1-score"],
                "support": class_report["support"],
            }
        )

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(metrics)

    # Plot metrics
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    df_plot = df.set_index("class")
    df_plot[["precision", "recall", "f1-score"]].plot(kind="bar", ax=plt.gca())
    plt.title("Precision, Recall, and F1-Score by Class")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    df.plot(kind="bar", x="class", y="support", ax=plt.gca())
    plt.title("Class Distribution in Test Set")
    plt.ylabel("Number of Samples")
    plt.xlabel("Class")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_metrics.png"), dpi=300)

    # Save metrics as JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "accuracy": float(results["accuracy"]),
                "precision": float(results["precision"]),
                "recall": float(results["recall"]),
                "f1": float(results["f1"]),
                "classification_report": results["classification_report"],
            },
            f,
            indent=4,
        )

    print(f"Visualizations saved to {output_dir}")


def main():
    """
    Main function for model evaluation and inference.
    """
    parser = argparse.ArgumentParser(description="Skin Cancer Classification Inference")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Directory to save results"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="marmal88/skin_cancer",
        help="Name of the HuggingFace dataset to evaluate on",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Size to resize images to"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--sample_images",
        type=int,
        default=5,
        help="Number of sample images to visualize per class",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess the dataset to add image_tensor field",
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Load model
    model, dx_categories, config = load_model(args.model_path, device=args.device)

    # Get image transforms
    transform = get_transforms(image_size=args.image_size)

    # Load dataset
    print(f"Loading {args.dataset_name} dataset, {args.split} split...")
    dataset = load_dataset(args.dataset_name)[args.split]

    # Check if we need to preprocess the dataset to match our training format
    if args.preprocess:
        dataset = preprocess_dataset(dataset, transform, dx_categories)
        has_preprocessed_tensors = True
    else:
        # Check if the dataset already has preprocessed tensors
        has_preprocessed_tensors = "image_tensor" in dataset[0]

    if has_preprocessed_tensors:
        print("Using preprocessed image tensors for evaluation")
    else:
        print("Using on-the-fly image transformation for evaluation")

    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(
        model,
        dataset,
        transform,
        device=args.device,
        dx_categories=dx_categories,
        batch_size=args.batch_size,
        preprocess_tensor_available=has_preprocessed_tensors,
    )

    # Visualize results
    print("Visualizing results...")
    visualize_results(results, dx_categories, output_dir=args.output_dir)

    # Sample inference on random images
    if args.sample_images > 0:
        print(
            f"Generating predictions for {args.sample_images} sample images per class..."
        )
        sample_output_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(sample_output_dir, exist_ok=True)

        # Group examples by class
        examples_by_class = {}
        for i, example in enumerate(dataset):
            label = example["label"]
            if label not in examples_by_class:
                examples_by_class[label] = []
            examples_by_class[label].append(i)

        # Sample images from each class
        for class_idx, indices in examples_by_class.items():
            # Get class name
            class_name = (
                dx_categories[class_idx] if dx_categories else f"Class {class_idx}"
            )

            # Create output directory for this class
            class_output_dir = os.path.join(sample_output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)

            # Sample random images from this class
            sample_count = min(args.sample_images, len(indices))
            sampled_indices = random.sample(indices, sample_count)

            for idx in sampled_indices:
                example = dataset[idx]

                # Save original image
                image = example["image"]
                image_id = example.get("image_id", f"img_{idx}")
                original_path = os.path.join(
                    class_output_dir, f"{image_id}_original.png"
                )
                image.save(original_path)

                # Make prediction
                prediction = predict_image(
                    original_path,
                    model,
                    transform,
                    device=args.device,
                    dx_categories=dx_categories,
                )

                # Create visualization with prediction
                plt.figure(figsize=(8, 6))
                plt.imshow(image)
                plt.axis("off")

                # Get ground truth and prediction
                true_label = class_name
                pred_label = prediction["class_label"]
                confidence = prediction["confidence"]

                # Set title color based on correctness
                color = "green" if true_label == pred_label else "red"

                plt.title(
                    f"True: {true_label}\nPredicted: {pred_label} ({confidence:.2f})",
                    color=color,
                    fontsize=12,
                )

                # Save visualization
                result_path = os.path.join(
                    class_output_dir, f"{image_id}_prediction.png"
                )
                plt.savefig(result_path, bbox_inches="tight", dpi=150)
                plt.close()

                # Also save prediction details as JSON
                metadata = {
                    "image_id": image_id,
                    "true_class": int(class_idx),
                    "true_label": true_label,
                    "predicted_class": prediction["predicted_class"],
                    "predicted_label": pred_label,
                    "confidence": float(confidence),
                    "probabilities": {
                        dx_categories[i] if dx_categories else f"Class {i}": float(p)
                        for i, p in enumerate(prediction["probabilities"])
                    },
                }

                with open(
                    os.path.join(class_output_dir, f"{image_id}_metadata.json"), "w"
                ) as f:
                    json.dump(metadata, f, indent=4)

    print(f"All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
