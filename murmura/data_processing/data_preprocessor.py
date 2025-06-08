import io
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Callable

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from numpy import dtype


class DataPreprocessor(ABC):
    """Abstract base class for data preprocessing strategies."""

    @abstractmethod
    def can_handle(self, data_sample: Any) -> bool:
        """Check if this preprocessor can handle the given data type."""
        pass

    @abstractmethod
    def preprocess(self, data: List[Any]) -> np.ndarray:
        """Preprocess the data into a numpy array."""
        pass


class ImageBytesPreprocessor(DataPreprocessor):
    """Preprocessor for image data stored as bytes (common in HuggingFace datasets)."""

    def __init__(
        self,
        target_mode: Optional[str] = None,
        normalize: bool = True,
        normalization_factor: float = 255.0,
        target_size: Optional[tuple] = None,
    ):
        """
        Initialize image bytes preprocessor.

        Args:
            target_mode: Target PIL mode ('RGB', 'L', etc.). If None, keeps original.
            normalize: Whether to normalize pixel values.
            normalization_factor: Factor to divide pixel values by.
            target_size: Target size as (width, height). If None, keeps original.
        """
        self.target_mode = target_mode
        self.normalize = normalize
        self.normalization_factor = normalization_factor
        self.target_size = target_size

    def can_handle(self, data_sample: Any) -> bool:
        """Check if data contains image bytes."""
        if isinstance(data_sample, dict):
            # Check for common image byte keys
            image_keys = ["bytes", "image_bytes", "data", "content"]
            for key in image_keys:
                if key in data_sample:
                    value = data_sample[key]
                    if isinstance(value, bytes):
                        # Try to detect if it's actually image data
                        return self._is_image_bytes(value)
        return False

    @staticmethod
    def _is_image_bytes(data: bytes) -> bool:
        """Check if bytes represent an image."""
        # Common image file signatures
        image_signatures = [
            b"\x89PNG",  # PNG
            b"\xff\xd8\xff",  # JPEG
            b"GIF8",  # GIF
            b"BM",  # BMP
            b"RIFF",  # WebP (contains WEBP later)
        ]
        return any(data.startswith(sig) for sig in image_signatures)

    def preprocess(self, data: List[Any]) -> np.ndarray:
        """Preprocess image bytes data with memory-efficient chunked processing."""
        import gc
        
        processed = []
        
        # Process images in smaller chunks to reduce memory usage
        chunk_size = min(32, len(data))  # Process max 32 images at once
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_processed = []
            
            for item in chunk:
                if isinstance(item, dict):
                    # Find the image bytes
                    image_bytes = None
                    for key in ["bytes", "image_bytes", "data", "content"]:
                        if key in item and isinstance(item[key], bytes):
                            image_bytes = item[key]
                            break

                    if image_bytes is None:
                        raise ValueError(
                            f"No image bytes found in dictionary with keys: {list(item.keys())}"
                        )

                    # Convert bytes to PIL Image
                    try:
                        img: PILImage = Image.open(io.BytesIO(image_bytes))

                        # Convert mode if specified
                        if self.target_mode and hasattr(img, "convert"):
                            img = img.convert(self.target_mode)

                        # Resize if specified
                        if self.target_size is not None and hasattr(img, "resize"):
                            img = img.resize(self.target_size)

                    except Exception as e:
                        raise ValueError(f"Failed to decode image bytes: {e}")
                else:
                    raise ValueError(
                        f"Expected dictionary with image bytes, got {type(item)}"
                    )

                # Convert to numpy with memory-efficient dtype
                img_array = np.array(img, dtype=np.uint8 if not self.normalize else np.float32)

                # Normalize if specified
                if self.normalize:
                    if img_array.dtype == np.uint8:
                        img_array = img_array.astype(np.float32) / self.normalization_factor
                    else:
                        img_array = img_array / self.normalization_factor

                chunk_processed.append(img_array)
                
                # Clean up PIL image to free memory
                img.close()
                del img
            
            processed.extend(chunk_processed)
            
            # Force garbage collection after each chunk
            if i > 0:  # Skip GC on first chunk to avoid overhead
                gc.collect()

        return np.array(processed)


class ImagePreprocessor(DataPreprocessor):
    """Preprocessor for PIL Image data with configurable parameters."""

    def __init__(
        self,
        target_mode: Optional[str] = None,
        normalize: bool = True,
        normalization_factor: float = 255.0,
        target_size: Optional[tuple] = None,
    ):
        """
        Initialize image preprocessor.

        Args:
            target_mode: Target PIL mode ('RGB', 'L', etc.). If None, keeps original.
            normalize: Whether to normalize pixel values.
            normalization_factor: Factor to divide pixel values by.
            target_size: Target size as (width, height). If None, keeps original.
        """
        self.target_mode = target_mode
        self.normalize = normalize
        self.normalization_factor = normalization_factor
        self.target_size = target_size

    def can_handle(self, data_sample: Any) -> bool:
        """Check if data is a PIL Image."""
        return hasattr(data_sample, "convert") or isinstance(data_sample, Image.Image)

    def preprocess(self, data: List[Any]) -> np.ndarray:
        """Preprocess PIL images with memory-efficient chunked processing."""
        import gc
        
        processed = []
        
        # Process images in smaller chunks to reduce memory usage
        chunk_size = min(32, len(data))  # Process max 32 images at once
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_processed = []
            
            for item in chunk:
                img: PILImage = item

                # Convert mode if specified
                if self.target_mode and hasattr(img, "convert"):
                    img = img.convert(self.target_mode)

                # Resize if specified
                if self.target_size is not None and hasattr(img, "resize"):
                    img = img.resize(self.target_size)

                # Convert to numpy with memory-efficient dtype
                img_array = np.array(img, dtype=np.uint8 if not self.normalize else np.float32)

                # Normalize if specified
                if self.normalize:
                    if img_array.dtype == np.uint8:
                        img_array = img_array.astype(np.float32) / self.normalization_factor
                    else:
                        img_array = img_array / self.normalization_factor

                chunk_processed.append(img_array)
            
            processed.extend(chunk_processed)
            
            # Force garbage collection after each chunk
            if i > 0:  # Skip GC on first chunk to avoid overhead
                gc.collect()

        return np.array(processed)


class DictPreprocessor(DataPreprocessor):
    """Preprocessor for dictionary data with configurable key extraction."""

    def __init__(
        self,
        key_extractors: Dict[str, Callable[[Any], np.ndarray]],
        primary_key: Optional[str] = None,
    ):
        """
        Initialize dictionary preprocessor.

        Args:
            key_extractors: Map of keys to extraction functions.
            primary_key: Primary key to extract if available.
        """
        self.key_extractors = key_extractors
        self.primary_key = primary_key

    def can_handle(self, data_sample: Any) -> bool:
        """Check if data is a dictionary."""
        return isinstance(data_sample, dict)

    def preprocess(self, data: List[Any]) -> np.ndarray:
        """Preprocess dictionary data."""
        processed = []

        for item in data:
            # Try primary key first
            if self.primary_key and self.primary_key in item:
                extractor = self.key_extractors.get(self.primary_key)
                if extractor:
                    processed.append(extractor(item[self.primary_key]))
                    continue

            # Try other keys
            extracted = False
            for key, extractor in self.key_extractors.items():
                if key in item:
                    processed.append(extractor(item[key]))
                    extracted = True
                    break

            if not extracted:
                raise ValueError(
                    f"Cannot extract data from dictionary with keys: {list(item.keys())}"
                )

        return np.array(processed)


class NumericPreprocessor(DataPreprocessor):
    """Preprocessor for numeric data."""

    def __init__(self, dtype: dtype[Any] = np.dtype(np.float32)):
        """
        Initialize numeric preprocessor.

        Args:
            dtype: Target numpy dtype.
        """
        self.dtype = dtype

    def can_handle(self, data_sample: Any) -> bool:
        """Check if data is numeric or can be converted to numeric."""
        try:
            np.array(data_sample, dtype=self.dtype)
            return True
        except (ValueError, TypeError):
            return False

    def preprocess(self, data: List[Any]) -> np.ndarray:
        """Preprocess numeric data."""
        return np.array(data, dtype=self.dtype)


class TensorPreprocessor(DataPreprocessor):
    """Preprocessor for tensor/array data."""

    def __init__(self, dtype: dtype[Any] = np.dtype(np.float32)):
        """
        Initialize tensor preprocessor.

        Args:
            dtype: Target numpy dtype.
        """
        self.dtype = dtype

    def can_handle(self, data_sample: Any) -> bool:
        """Check if data is already a tensor/array."""
        return isinstance(data_sample, np.ndarray) or hasattr(data_sample, "numpy")

    def preprocess(self, data: List[Any]) -> np.ndarray:
        """Preprocess tensor data."""
        processed = []
        for item in data:
            if hasattr(item, "numpy"):
                # PyTorch/TensorFlow tensor
                processed.append(item.numpy().astype(self.dtype))
            elif isinstance(item, np.ndarray):
                processed.append(item.astype(self.dtype))
            else:
                processed.append(np.array(item, dtype=self.dtype))

        return np.array(processed)


class GenericDataPreprocessor:
    """
    Generic data preprocessor that automatically detects data types and applies
    appropriate preprocessing strategies.
    """

    def __init__(self, preprocessors: Optional[List[DataPreprocessor]] = None):
        """
        Initialize generic preprocessor.

        Args:
            preprocessors: List of preprocessing strategies. If None, uses defaults.
        """
        if preprocessors is None:
            # Default preprocessors for common data types - order matters!
            self.preprocessors = [
                TensorPreprocessor(),  # Try tensors first
                ImageBytesPreprocessor(),  # Then image bytes (for HuggingFace datasets)
                ImagePreprocessor(),  # Then direct PIL images
                DictPreprocessor(  # Then dictionaries with custom extractors
                    key_extractors={
                        "image": lambda x: self._preprocess_image_from_dict(x),
                        "pixel_values": lambda x: np.array(x, dtype=np.float32),
                        "data": lambda x: np.array(x, dtype=np.float32),
                        "features": lambda x: np.array(x, dtype=np.float32),
                    },
                    primary_key="image",
                ),
                NumericPreprocessor(),  # Finally, try numeric conversion
            ]
        else:
            self.preprocessors = preprocessors

    @staticmethod
    def _preprocess_image_from_dict(img_data: Any) -> np.ndarray:
        """Helper to preprocess image data extracted from dictionary."""
        if hasattr(img_data, "convert") or isinstance(img_data, Image.Image):
            return np.array(img_data, dtype=np.float32) / 255.0
        elif isinstance(img_data, bytes):
            # Handle image bytes
            try:
                img = Image.open(io.BytesIO(img_data))
                return np.array(img, dtype=np.float32) / 255.0
            except Exception:
                raise ValueError("Failed to decode image bytes")
        else:
            return np.array(img_data, dtype=np.float32)

    def preprocess_features(self, feature_data: List[Any]) -> np.ndarray:
        """
        Automatically detect data type and preprocess features.

        Args:
            feature_data: List of feature data in any format.

        Returns:
            Preprocessed numpy array.

        Raises:
            ValueError: If no preprocessor can handle the data type.
        """
        if not feature_data:
            return np.array([])

        # Get a sample to determine data type
        sample = feature_data[0]

        # Find appropriate preprocessor
        for preprocessor in self.preprocessors:
            if preprocessor.can_handle(sample):
                try:
                    return preprocessor.preprocess(feature_data)
                except Exception:
                    # If this preprocessor fails, try the next one
                    continue

        # If no preprocessor worked, raise an error with helpful info
        sample_str = (
            str(sample)
            if not hasattr(sample, "__len__") or len(str(sample)) < 100
            else str(sample)[:100] + "..."
        )
        raise ValueError(
            f"Cannot preprocess data of type {type(sample)}. Sample value: {sample_str}"
        )

    def add_preprocessor(
        self, preprocessor: DataPreprocessor, priority: int = -1
    ) -> None:
        """
        Add a custom preprocessor.

        Args:
            preprocessor: The preprocessor to add.
            priority: Position in the list (-1 for end, 0 for beginning).
        """
        if priority == -1:
            self.preprocessors.append(preprocessor)
        else:
            self.preprocessors.insert(priority, preprocessor)


# Dataset-specific factory functions for common use cases
def create_image_preprocessor(
    grayscale: bool = False, normalize: bool = True, target_size: Optional[tuple] = None
) -> GenericDataPreprocessor:
    """Create preprocessor optimized for image datasets."""
    target_mode = "L" if grayscale else None

    preprocessors = [
        TensorPreprocessor(),
        ImageBytesPreprocessor(  # Add support for image bytes
            target_mode=target_mode, normalize=normalize, target_size=target_size
        ),
        ImagePreprocessor(
            target_mode=target_mode, normalize=normalize, target_size=target_size
        ),
        DictPreprocessor(
            key_extractors={
                "image": lambda x: _process_dict_image(x, target_mode, normalize),
                "pixel_values": lambda x: np.array(x, dtype=np.float32),
                "bytes": lambda x: _process_image_bytes(
                    x, target_mode, normalize, target_size
                ),
            },
            primary_key="image",
        ),
        NumericPreprocessor(),
    ]

    return GenericDataPreprocessor(preprocessors)


def create_text_preprocessor() -> GenericDataPreprocessor:
    """Create preprocessor optimized for text datasets."""
    preprocessors = [
        TensorPreprocessor(),
        DictPreprocessor(
            key_extractors={
                "input_ids": lambda x: np.array(x, dtype=np.int64),
                "attention_mask": lambda x: np.array(x, dtype=np.int64),
                "text": lambda x: _encode_text(x),
                "tokens": lambda x: np.array(x, dtype=np.int64),
            },
            primary_key="input_ids",
        ),
        NumericPreprocessor(dtype=np.dtype(np.int64)),
    ]

    return GenericDataPreprocessor(preprocessors)


def create_tabular_preprocessor() -> GenericDataPreprocessor:
    """Create preprocessor optimized for tabular datasets."""
    preprocessors = [
        TensorPreprocessor(),
        NumericPreprocessor(),
        DictPreprocessor(
            key_extractors={
                "features": lambda x: np.array(x, dtype=np.float32),
                "data": lambda x: np.array(x, dtype=np.float32),
            }
        ),
    ]

    return GenericDataPreprocessor(preprocessors)


# Helper functions
def _process_dict_image(
    img_data: Any, target_mode: Optional[str], normalize: bool
) -> np.ndarray:
    """Process image data extracted from dictionary."""
    if hasattr(img_data, "convert") or isinstance(img_data, Image.Image):
        img: PILImage = img_data
        if target_mode and hasattr(img, "convert"):
            img = img.convert(target_mode)
        img_array = np.array(img, dtype=np.float32)
        if normalize:
            img_array = img_array / 255.0
        return img_array
    elif isinstance(img_data, bytes):
        return _process_image_bytes(img_data, target_mode, normalize, None)
    else:
        return np.array(img_data, dtype=np.float32)


def _process_image_bytes(
    img_bytes: bytes,
    target_mode: Optional[str],
    normalize: bool,
    target_size: Optional[tuple],
) -> np.ndarray:
    """Process image bytes into numpy array."""
    try:
        img: PILImage = Image.open(io.BytesIO(img_bytes))

        if target_mode and hasattr(img, "convert"):
            img = img.convert(target_mode)

        if target_size is not None and hasattr(img, "resize"):
            img = img.resize(target_size)

        img_array = np.array(img, dtype=np.float32)

        if normalize:
            img_array = img_array / 255.0

        return img_array
    except Exception as e:
        raise ValueError(f"Failed to process image bytes: {e}")


def _encode_text(text: str) -> np.ndarray:
    """Simple text encoding (you'd replace this with your tokenizer)."""
    # This is a placeholder - you'd use your actual tokenizer here
    return np.array(
        [ord(c) for c in text[:100]], dtype=np.int64
    )  # Simple char encoding
