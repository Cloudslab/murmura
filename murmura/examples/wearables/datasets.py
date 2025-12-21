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


class PPGDaLiADataset(Dataset):
    """PPG-DaLiA (PPG Dataset for Activity Recognition with Wearables).

    Dataset from 15 subjects performing 8 activities while wearing
    wrist-worn (Empatica E4) and chest-worn (RespiBAN) sensors.

    Activities:
        0: transient/no activity, 1: sitting, 2: ascending stairs,
        3: descending stairs, 4: walking, 5: cycling, 6: driving,
        7: table soccer

    Reference:
        Attila Reiss et al. "Deep PPG: Large-Scale Heart Rate Estimation
        with Convolutional Neural Networks." Sensors 2019.
    """

    ACTIVITY_LABELS = [1, 2, 3, 4, 5, 6, 7]  # Exclude 0 (transient)
    ACTIVITY_NAMES = {
        0: "transient",
        1: "sitting",
        2: "ascending_stairs",
        3: "descending_stairs",
        4: "walking",
        5: "cycling",
        6: "driving",
        7: "table_soccer",
    }

    # Wrist sensor sampling rates (Empatica E4)
    WRIST_ACC_HZ = 32
    WRIST_BVP_HZ = 64
    WRIST_EDA_HZ = 4
    WRIST_TEMP_HZ = 4

    # Activity labels are at 4 Hz (0.25 sec per sample)
    ACTIVITY_HZ = 4

    def __init__(
        self,
        root: str,
        subjects: Optional[List[int]] = None,
        activities: Optional[List[int]] = None,
        window_size: int = 32,  # 8 seconds at 4 Hz
        window_stride: int = 16,  # 4 second stride
        normalize: bool = True,
        use_wrist_only: bool = True,  # Use only wrist sensors for simplicity
    ):
        """Initialize PPG-DaLiA dataset.

        Args:
            root: Path to 'PPG_FieldStudy' directory
            subjects: List of subject IDs to include (default: all 1-15)
            activities: List of activity IDs to include (default: 1-7)
            window_size: Sliding window size in samples at 4 Hz
            window_stride: Stride between windows
            normalize: Whether to normalize features
            use_wrist_only: Use only wrist sensors (simpler, fewer features)
        """
        self.root = Path(root)
        self.subjects = subjects or list(range(1, 16))  # S1-S15
        self.activities = activities or self.ACTIVITY_LABELS
        self.window_size = window_size
        self.window_stride = window_stride
        self.normalize = normalize
        self.use_wrist_only = use_wrist_only

        # Create activity to index mapping
        self.activity_to_idx = {a: i for i, a in enumerate(self.activities)}

        self._load_data()

    def _load_data(self) -> None:
        """Load and preprocess data from all subjects."""
        import pickle

        all_features = []
        all_labels = []
        all_subjects = []

        for subject_id in self.subjects:
            pkl_path = self.root / f"S{subject_id}" / f"S{subject_id}.pkl"
            if not pkl_path.exists():
                continue

            with open(pkl_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')

            # Extract features from wrist sensors at 4 Hz (EDA and TEMP native rate)
            features = self._extract_features(data)
            activities = data['activity'].flatten().astype(int)

            # Ensure features and activities align
            min_len = min(len(features), len(activities))
            features = features[:min_len]
            activities = activities[:min_len]

            # Filter by valid activities
            valid_mask = np.isin(activities, self.activities)
            features = features[valid_mask]
            activities = activities[valid_mask]

            if len(features) == 0:
                continue

            # Create sliding windows
            windows, labels, subjects = self._create_windows(
                features, activities, subject_id
            )

            if len(windows) > 0:
                all_features.append(windows)
                all_labels.append(labels)
                all_subjects.append(subjects)

        if not all_features:
            raise ValueError(f"No valid data found in {self.root}")

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

    def _extract_features(self, data: Dict) -> np.ndarray:
        """Extract and downsample features from sensor data.

        Returns features at 4 Hz (matching activity label rate).
        """
        wrist = data['signal']['wrist']

        # Get raw signals
        eda = wrist['EDA'].flatten()  # Already at 4 Hz
        temp = wrist['TEMP'].flatten()  # Already at 4 Hz

        # Downsample ACC from 32 Hz to 4 Hz (take every 8th sample)
        acc = wrist['ACC']  # Shape: (N*8, 3)
        acc_downsampled = acc[::8, :]  # Downsample to 4 Hz

        # Downsample BVP from 64 Hz to 4 Hz (take every 16th sample)
        bvp = wrist['BVP'].flatten()
        bvp_downsampled = bvp[::16]

        # Align all signals to minimum length
        min_len = min(len(eda), len(temp), len(acc_downsampled), len(bvp_downsampled))
        eda = eda[:min_len]
        temp = temp[:min_len]
        acc_downsampled = acc_downsampled[:min_len]
        bvp_downsampled = bvp_downsampled[:min_len]

        # Stack features: [EDA, TEMP, ACC_x, ACC_y, ACC_z, BVP] = 6 features
        features = np.column_stack([
            eda,
            temp,
            acc_downsampled,
            bvp_downsampled,
        ])

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)

        return features.astype(np.float32)

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
