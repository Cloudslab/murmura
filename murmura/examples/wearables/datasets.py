"""PyTorch Dataset classes for wearable sensor datasets."""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class UCIHARDataset(Dataset):
    """UCI Human Activity Recognition Dataset.

    Dataset collected from 30 subjects performing 6 activities while wearing
    a smartphone (Samsung Galaxy S II) on the waist.

    Activities:
        1: WALKING, 2: WALKING_UPSTAIRS, 3: WALKING_DOWNSTAIRS,
        4: SITTING, 5: STANDING, 6: LAYING

    Reference:
        Davide Anguita, et al. "A Public Domain Dataset for Human Activity
        Recognition Using Smartphones." ESANN 2013.
    """

    ACTIVITIES = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
    }
    NUM_FEATURES = 561
    NUM_CLASSES = 6

    def __init__(
        self,
        root: str,
        split: str = "train",
        normalize: bool = True,
    ):
        """Initialize UCI HAR dataset.

        Args:
            root: Path to 'UCI HAR Dataset' directory
            split: 'train' or 'test'
            normalize: Whether to normalize features (already normalized in dataset)
        """
        self.root = Path(root)
        self.split = split
        self.normalize = normalize

        self._load_data()

    def _load_data(self) -> None:
        """Load data from text files."""
        split_dir = self.root / self.split

        # Load features
        x_file = split_dir / f"X_{self.split}.txt"
        self.features = np.loadtxt(x_file)

        # Load labels (1-indexed, convert to 0-indexed)
        y_file = split_dir / f"y_{self.split}.txt"
        self.labels = np.loadtxt(y_file, dtype=np.int64) - 1

        # Load subject IDs
        subject_file = split_dir / f"subject_{self.split}.txt"
        self.subjects = np.loadtxt(subject_file, dtype=np.int64)

        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array for partitioning."""
        return self.labels.numpy()

    def get_subjects(self) -> np.ndarray:
        """Get all subject IDs as numpy array for natural partitioning."""
        return self.subjects


class PAMAP2Dataset(Dataset):
    """PAMAP2 Physical Activity Monitoring Dataset.

    Dataset from 9 subjects performing 18 different physical activities
    with 3 IMUs (hand, chest, ankle) and a heart rate monitor.

    Activities (commonly used subset):
        1: Lying, 2: Sitting, 3: Standing, 4: Walking, 5: Running,
        6: Cycling, 7: Nordic walking, 12: Ascending stairs,
        13: Descending stairs, 16: Vacuum cleaning, 17: Ironing, 24: Rope jumping

    Reference:
        Attila Reiss and Didier Stricker. "Introducing a New Benchmarked Dataset
        for Activity Monitoring." ISWC 2012.
    """

    # Commonly used activity labels (excluding transient label 0)
    ACTIVITY_LABELS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    ACTIVITY_NAMES = {
        1: "lying", 2: "sitting", 3: "standing", 4: "walking",
        5: "running", 6: "cycling", 7: "nordic_walking",
        12: "ascending_stairs", 13: "descending_stairs",
        16: "vacuum_cleaning", 17: "ironing", 24: "rope_jumping",
    }

    # Column indices
    TIMESTAMP_COL = 0
    ACTIVITY_COL = 1
    HEART_RATE_COL = 2
    IMU_HAND_START = 3
    IMU_CHEST_START = 20
    IMU_ANKLE_START = 37
    TOTAL_COLS = 54

    # Feature columns (excluding timestamp, activity, and orientation columns)
    # Each IMU has 17 cols: temp(1) + accel_16g(3) + accel_6g(3) + gyro(3) + mag(3) + orient(4)
    # We exclude orientation (4 cols per IMU) as noted invalid in documentation
    FEATURE_COLS_PER_IMU = 13  # Excluding orientation

    def __init__(
        self,
        root: str,
        subjects: Optional[List[int]] = None,
        activities: Optional[List[int]] = None,
        window_size: int = 100,
        window_stride: int = 50,
        normalize: bool = True,
        include_heart_rate: bool = True,
    ):
        """Initialize PAMAP2 dataset.

        Args:
            root: Path to 'PAMAP2_Dataset' directory
            subjects: List of subject IDs to include (default: all 101-109)
            activities: List of activity IDs to include (default: common 12)
            window_size: Sliding window size in samples (100 samples = 1 sec at 100Hz)
            window_stride: Stride between windows
            normalize: Whether to normalize features
            include_heart_rate: Whether to include heart rate feature
        """
        self.root = Path(root)
        self.subjects = subjects or list(range(101, 110))
        self.activities = activities or self.ACTIVITY_LABELS
        self.window_size = window_size
        self.window_stride = window_stride
        self.normalize = normalize
        self.include_heart_rate = include_heart_rate

        # Create activity to index mapping
        self.activity_to_idx = {a: i for i, a in enumerate(self.activities)}

        self._load_data()

    def _get_feature_columns(self) -> List[int]:
        """Get indices of feature columns to use."""
        cols = []

        if self.include_heart_rate:
            cols.append(self.HEART_RATE_COL)

        # For each IMU, get temperature + accelerometer + gyroscope + magnetometer
        for imu_start in [self.IMU_HAND_START, self.IMU_CHEST_START, self.IMU_ANKLE_START]:
            # Temperature (1), Accel 16g (3), Accel 6g (3), Gyro (3), Mag (3) = 13 cols
            cols.extend(range(imu_start, imu_start + 13))

        return cols

    def _load_data(self) -> None:
        """Load and preprocess data from all subjects."""
        feature_cols = self._get_feature_columns()
        all_features = []
        all_labels = []
        all_subjects = []

        protocol_dir = self.root / "Protocol"

        for subject_id in self.subjects:
            filepath = protocol_dir / f"subject{subject_id}.dat"
            if not filepath.exists():
                continue

            # Load data
            data = np.loadtxt(filepath)

            # Filter by activity (exclude transient label 0 and unwanted activities)
            activity_col = data[:, self.ACTIVITY_COL].astype(int)
            valid_mask = np.isin(activity_col, self.activities)
            data = data[valid_mask]
            activity_col = activity_col[valid_mask]

            # Extract features
            features = data[:, feature_cols]

            # Handle NaN values by interpolation or zero-fill
            features = self._handle_nan(features)

            # Create sliding windows
            windows, labels, subjects = self._create_windows(
                features, activity_col, subject_id
            )

            all_features.append(windows)
            all_labels.append(labels)
            all_subjects.append(subjects)

        # Concatenate all data
        self.features = np.vstack(all_features)
        self.labels = np.concatenate(all_labels)
        self.subject_ids = np.concatenate(all_subjects)

        # Normalize if requested
        if self.normalize:
            self._normalize_features()

        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _handle_nan(self, features: np.ndarray) -> np.ndarray:
        """Handle NaN values in features."""
        # Replace NaN with column mean, or 0 if all NaN
        for col in range(features.shape[1]):
            nan_mask = np.isnan(features[:, col])
            if nan_mask.any():
                valid_vals = features[~nan_mask, col]
                if len(valid_vals) > 0:
                    features[nan_mask, col] = np.mean(valid_vals)
                else:
                    features[nan_mask, col] = 0.0
        return features

    def _create_windows(
        self, features: np.ndarray, activities: np.ndarray, subject_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sliding windows from continuous data."""
        windows = []
        labels = []
        subjects = []

        num_samples = len(features)
        for start in range(0, num_samples - self.window_size + 1, self.window_stride):
            end = start + self.window_size
            window = features[start:end]

            # Use majority activity in window as label
            window_activities = activities[start:end]
            activity, counts = np.unique(window_activities, return_counts=True)
            majority_activity = activity[np.argmax(counts)]

            if majority_activity in self.activity_to_idx:
                # Flatten window: (window_size, num_features) -> (window_size * num_features,)
                windows.append(window.flatten())
                labels.append(self.activity_to_idx[majority_activity])
                subjects.append(subject_id)

        if not windows:
            return np.array([]), np.array([]), np.array([])

        return np.array(windows), np.array(labels), np.array(subjects)

    def _normalize_features(self) -> None:
        """Normalize features to zero mean and unit variance."""
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0)
        self.std[self.std == 0] = 1  # Avoid division by zero
        self.features = (self.features - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array for partitioning."""
        return self.labels.numpy()

    def get_subjects(self) -> np.ndarray:
        """Get all subject IDs as numpy array for natural partitioning."""
        return self.subject_ids

    @property
    def num_features(self) -> int:
        """Get number of features per sample."""
        return self.features.shape[1]

    @property
    def num_classes(self) -> int:
        """Get number of activity classes."""
        return len(self.activities)


class ExtraSensoryDataset(Dataset):
    """ExtraSensory Mobile Sensing Dataset.

    Dataset from multiple users with smartphone and smartwatch sensors,
    featuring multi-label activity, context, and location annotations.

    Reference:
        Yonatan Vaizman, et al. "Recognizing Detailed Human Context In-the-Wild
        from Smartphones and Smartwatches." IEEE Pervasive Computing 2017.
    """

    # Activity labels for single-label classification
    ACTIVITY_LABELS = [
        "LYING_DOWN", "SITTING", "FIX_walking", "FIX_running",
        "BICYCLING", "SLEEPING", "STROLLING",
    ]

    def __init__(
        self,
        root: str,
        users: Optional[List[str]] = None,
        target_label: str = "FIX_walking",  # Primary label for single-label mode
        multi_label: bool = False,
        label_columns: Optional[List[str]] = None,
        normalize: bool = True,
        handle_missing: str = "zero",  # 'zero', 'mean', or 'drop'
    ):
        """Initialize ExtraSensory dataset.

        Args:
            root: Path to 'ExtraSensory' directory containing user CSV files
            users: List of user file prefixes to include (e.g., ['user1', 'user2'])
            target_label: Column name for single-label classification
            multi_label: Whether to use multi-label classification
            label_columns: List of label columns for multi-label mode
            normalize: Whether to normalize features
            handle_missing: How to handle missing values ('zero', 'mean', 'drop')
        """
        self.root = Path(root)
        self.target_label = target_label
        self.multi_label = multi_label
        self.normalize = normalize
        self.handle_missing = handle_missing

        # Find all user files
        self._discover_users(users)

        # Discover label columns
        self._discover_labels(label_columns)

        self._load_data()

    def _discover_users(self, users: Optional[List[str]]) -> None:
        """Discover user CSV files in the root directory."""
        if users is not None:
            self.user_files = [
                self.root / f"{u}.features_labels.csv" for u in users
            ]
        else:
            self.user_files = sorted(self.root.glob("*.features_labels.csv"))

        if not self.user_files:
            raise ValueError(f"No user CSV files found in {self.root}")

    def _discover_labels(self, label_columns: Optional[List[str]]) -> None:
        """Discover available label columns from first file."""
        import pandas as pd

        # Read first file to get column names
        sample_df = pd.read_csv(self.user_files[0], nrows=1)

        # Label columns are those that start with "label:"
        metadata_cols = {"timestamp", "label_source"}
        self.all_columns = list(sample_df.columns)

        # Identify label columns (binary labels from the dataset)
        self.available_labels = [
            col for col in self.all_columns
            if col.startswith("label:") and col not in metadata_cols
        ]

        if label_columns is not None:
            self.label_columns = label_columns
        else:
            self.label_columns = self.available_labels

        # Feature columns are everything else (not labels, not metadata)
        self.feature_columns = [
            col for col in self.all_columns
            if not col.startswith("label:") and col not in metadata_cols
        ]

        # Resolve target_label: add "label:" prefix if needed
        if not self.target_label.startswith("label:"):
            self.target_label = f"label:{self.target_label}"

    def _load_data(self) -> None:
        """Load and preprocess data from all users."""
        import pandas as pd

        all_features = []
        all_labels = []
        all_users = []

        for user_idx, user_file in enumerate(self.user_files):
            df = pd.read_csv(user_file)

            # Extract features
            features = df[self.feature_columns].values.astype(np.float32)

            # Handle missing values
            features = self._handle_missing_values(features)

            if self.multi_label:
                # Multi-label: stack selected label columns
                labels = df[self.label_columns].values.astype(np.float32)
                labels = np.nan_to_num(labels, nan=0.0)  # Convert NaN to 0
            else:
                # Single-label: use target label column
                if self.target_label not in df.columns:
                    raise ValueError(f"Target label '{self.target_label}' not found")
                labels = df[self.target_label].values.astype(np.float32)
                labels = np.nan_to_num(labels, nan=0.0)

            user_ids = np.full(len(features), user_idx)

            all_features.append(features)
            all_labels.append(labels)
            all_users.append(user_ids)

        # Concatenate all data
        self.features = np.vstack(all_features)
        self.labels = np.vstack(all_labels) if self.multi_label else np.concatenate(all_labels)
        self.user_ids = np.concatenate(all_users)

        # Normalize if requested
        if self.normalize:
            self._normalize_features()

        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        if self.multi_label:
            self.labels = torch.tensor(self.labels, dtype=torch.float32)
        else:
            self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing/NaN values in features."""
        if self.handle_missing == "zero":
            return np.nan_to_num(features, nan=0.0)
        elif self.handle_missing == "mean":
            col_means = np.nanmean(features, axis=0)
            col_means = np.nan_to_num(col_means, nan=0.0)
            inds = np.where(np.isnan(features))
            features[inds] = np.take(col_means, inds[1])
            return features
        elif self.handle_missing == "drop":
            # Mark rows with any NaN for later filtering
            valid_rows = ~np.any(np.isnan(features), axis=1)
            return features[valid_rows]
        else:
            raise ValueError(f"Unknown handle_missing mode: {self.handle_missing}")

    def _normalize_features(self) -> None:
        """Normalize features to zero mean and unit variance."""
        self.mean = np.nanmean(self.features, axis=0)
        self.std = np.nanstd(self.features, axis=0)
        self.std[self.std == 0] = 1  # Avoid division by zero
        self.features = (self.features - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array for partitioning.

        For multi-label, returns the first label column for partitioning purposes.
        """
        labels = self.labels.numpy() if isinstance(self.labels, torch.Tensor) else self.labels
        if self.multi_label:
            return labels[:, 0].astype(np.int64)
        return labels.astype(np.int64)

    def get_users(self) -> np.ndarray:
        """Get all user IDs as numpy array for natural partitioning."""
        return self.user_ids

    @property
    def num_features(self) -> int:
        """Get number of features per sample."""
        return self.features.shape[1]

    @property
    def num_classes(self) -> int:
        """Get number of classes/labels."""
        if self.multi_label:
            return len(self.label_columns)
        return 2  # Binary classification for single label
