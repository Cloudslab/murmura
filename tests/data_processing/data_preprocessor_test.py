"""
Tests for data preprocessing functionality.
"""
import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from murmura.data_processing.data_preprocessor import (
    DataPreprocessor,
    ImageBytesPreprocessor,
    ImagePreprocessor,
    DictPreprocessor,
    NumericPreprocessor,
    TensorPreprocessor,
    GenericDataPreprocessor,
    create_image_preprocessor,
    create_text_preprocessor,
    create_tabular_preprocessor,
    _process_dict_image,
    _process_image_bytes,
    _encode_text,
)


class TestImageBytesPreprocessor:
    """Test ImageBytesPreprocessor functionality."""

    @pytest.fixture
    def image_bytes(self):
        """Create test image bytes."""
        # Create a simple test image
        img = Image.new('RGB', (10, 10), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()

    @pytest.fixture
    def preprocessor(self):
        """Create ImageBytesPreprocessor instance."""
        return ImageBytesPreprocessor()

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        preprocessor = ImageBytesPreprocessor()
        assert preprocessor.target_mode is None
        assert preprocessor.normalize is True
        assert preprocessor.normalization_factor == 255.0
        assert preprocessor.target_size is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        preprocessor = ImageBytesPreprocessor(
            target_mode='L',
            normalize=False,
            normalization_factor=1.0,
            target_size=(32, 32)
        )
        assert preprocessor.target_mode == 'L'
        assert preprocessor.normalize is False
        assert preprocessor.normalization_factor == 1.0
        assert preprocessor.target_size == (32, 32)

    def test_can_handle_image_bytes_dict(self, preprocessor, image_bytes):
        """Test can_handle with dictionary containing image bytes."""
        data = {"bytes": image_bytes}
        assert preprocessor.can_handle(data) is True

    def test_can_handle_non_image_bytes(self, preprocessor):
        """Test can_handle with non-image bytes."""
        data = {"bytes": b"not an image"}
        assert preprocessor.can_handle(data) is False

    def test_can_handle_no_bytes_key(self, preprocessor):
        """Test can_handle with dictionary without bytes key."""
        data = {"other": "value"}
        assert preprocessor.can_handle(data) is False

    def test_can_handle_non_dict(self, preprocessor):
        """Test can_handle with non-dictionary data."""
        assert preprocessor.can_handle("string") is False
        assert preprocessor.can_handle(123) is False

    def test_is_image_bytes_png(self):
        """Test _is_image_bytes with PNG signature."""
        png_bytes = b"\x89PNG\r\n\x1a\n"
        assert ImageBytesPreprocessor._is_image_bytes(png_bytes) is True

    def test_is_image_bytes_jpeg(self):
        """Test _is_image_bytes with JPEG signature."""
        jpeg_bytes = b"\xff\xd8\xff\xe0"
        assert ImageBytesPreprocessor._is_image_bytes(jpeg_bytes) is True

    def test_is_image_bytes_invalid(self):
        """Test _is_image_bytes with invalid bytes."""
        invalid_bytes = b"not an image"
        assert ImageBytesPreprocessor._is_image_bytes(invalid_bytes) is False

    def test_preprocess_basic(self, preprocessor, image_bytes):
        """Test basic preprocessing of image bytes."""
        data = [{"bytes": image_bytes}]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert result[0].dtype == np.float32
        assert np.all(result[0] >= 0.0) and np.all(result[0] <= 1.0)  # Normalized

    def test_preprocess_multiple_images(self, image_bytes):
        """Test preprocessing multiple images."""
        preprocessor = ImageBytesPreprocessor()
        data = [{"bytes": image_bytes}, {"image_bytes": image_bytes}]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_preprocess_with_mode_conversion(self, image_bytes):
        """Test preprocessing with mode conversion."""
        preprocessor = ImageBytesPreprocessor(target_mode='L')
        data = [{"bytes": image_bytes}]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result[0].shape) == 2  # Grayscale should be 2D

    def test_preprocess_without_normalization(self, image_bytes):
        """Test preprocessing without normalization."""
        preprocessor = ImageBytesPreprocessor(normalize=False)
        data = [{"bytes": image_bytes}]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        assert np.any(result[0] > 1.0)  # Should have values > 1 if not normalized

    def test_preprocess_with_resize(self, image_bytes):
        """Test preprocessing with resizing."""
        preprocessor = ImageBytesPreprocessor(target_size=(5, 5))
        data = [{"bytes": image_bytes}]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        assert result[0].shape[:2] == (5, 5)

    def test_preprocess_missing_bytes_key(self, preprocessor):
        """Test preprocessing with missing bytes key."""
        data = [{"other": "value"}]
        with pytest.raises(ValueError, match="No image bytes found"):
            preprocessor.preprocess(data)

    def test_preprocess_invalid_image_bytes(self, preprocessor):
        """Test preprocessing with invalid image bytes."""
        data = [{"bytes": b"invalid image data"}]
        with pytest.raises(ValueError, match="Failed to decode image bytes"):
            preprocessor.preprocess(data)

    def test_preprocess_non_dict_item(self, preprocessor):
        """Test preprocessing with non-dictionary item."""
        data = ["not a dict"]
        with pytest.raises(ValueError, match="Expected dictionary with image bytes"):
            preprocessor.preprocess(data)


class TestImagePreprocessor:
    """Test ImagePreprocessor functionality."""

    @pytest.fixture
    def pil_image(self):
        """Create test PIL image."""
        return Image.new('RGB', (10, 10), color='blue')

    @pytest.fixture
    def preprocessor(self):
        """Create ImagePreprocessor instance."""
        return ImagePreprocessor()

    def test_can_handle_pil_image(self, preprocessor, pil_image):
        """Test can_handle with PIL Image."""
        assert preprocessor.can_handle(pil_image) is True

    def test_can_handle_non_image(self, preprocessor):
        """Test can_handle with non-image data."""
        assert preprocessor.can_handle("string") is False
        assert preprocessor.can_handle(123) is False

    def test_preprocess_basic(self, preprocessor, pil_image):
        """Test basic preprocessing of PIL images."""
        data = [pil_image]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert result[0].dtype == np.float32

    def test_preprocess_with_mode_conversion(self, pil_image):
        """Test preprocessing with mode conversion."""
        preprocessor = ImagePreprocessor(target_mode='L')
        data = [pil_image]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        assert len(result[0].shape) == 2  # Grayscale should be 2D


class TestDictPreprocessor:
    """Test DictPreprocessor functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create DictPreprocessor instance."""
        key_extractors = {
            "data": lambda x: np.array(x, dtype=np.float32),
            "values": lambda x: np.array(x, dtype=np.float32),
        }
        return DictPreprocessor(key_extractors, primary_key="data")

    def test_can_handle_dict(self, preprocessor):
        """Test can_handle with dictionary."""
        assert preprocessor.can_handle({"key": "value"}) is True

    def test_can_handle_non_dict(self, preprocessor):
        """Test can_handle with non-dictionary."""
        assert preprocessor.can_handle("string") is False
        assert preprocessor.can_handle([1, 2, 3]) is False

    def test_preprocess_with_primary_key(self, preprocessor):
        """Test preprocessing using primary key."""
        data = [{"data": [1, 2, 3], "other": [4, 5, 6]}]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result[0], np.array([1, 2, 3], dtype=np.float32))

    def test_preprocess_without_primary_key(self, preprocessor):
        """Test preprocessing without primary key."""
        data = [{"values": [1, 2, 3]}]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result[0], np.array([1, 2, 3], dtype=np.float32))

    def test_preprocess_no_extractable_keys(self, preprocessor):
        """Test preprocessing with no extractable keys."""
        data = [{"unknown": [1, 2, 3]}]
        with pytest.raises(ValueError, match="Cannot extract data from dictionary"):
            preprocessor.preprocess(data)


class TestNumericPreprocessor:
    """Test NumericPreprocessor functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create NumericPreprocessor instance."""
        return NumericPreprocessor()

    def test_can_handle_numeric(self, preprocessor):
        """Test can_handle with numeric data."""
        assert preprocessor.can_handle(123) is True
        assert preprocessor.can_handle(1.5) is True
        assert preprocessor.can_handle([1, 2, 3]) is True

    def test_can_handle_non_numeric(self, preprocessor):
        """Test can_handle with non-numeric data."""
        assert preprocessor.can_handle("string") is False

    def test_preprocess_numeric_list(self, preprocessor):
        """Test preprocessing numeric list."""
        data = [[1, 2, 3], [4, 5, 6]]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))


class TestTensorPreprocessor:
    """Test TensorPreprocessor functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create TensorPreprocessor instance."""
        return TensorPreprocessor()

    def test_can_handle_numpy_array(self, preprocessor):
        """Test can_handle with numpy array."""
        arr = np.array([1, 2, 3])
        assert preprocessor.can_handle(arr) is True

    def test_can_handle_tensor_like(self, preprocessor):
        """Test can_handle with tensor-like object."""
        mock_tensor = Mock()
        mock_tensor.numpy = Mock()
        assert preprocessor.can_handle(mock_tensor) is True

    def test_can_handle_non_tensor(self, preprocessor):
        """Test can_handle with non-tensor data."""
        assert preprocessor.can_handle("string") is False

    def test_preprocess_numpy_arrays(self, preprocessor):
        """Test preprocessing numpy arrays."""
        data = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_preprocess_tensor_like(self, preprocessor):
        """Test preprocessing tensor-like objects."""
        mock_tensor = Mock()
        mock_tensor.numpy.return_value = np.array([1, 2, 3])
        
        data = [mock_tensor]
        result = preprocessor.preprocess(data)
        
        assert isinstance(result, np.ndarray)
        mock_tensor.numpy.assert_called_once()


class TestGenericDataPreprocessor:
    """Test GenericDataPreprocessor functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create GenericDataPreprocessor instance."""
        return GenericDataPreprocessor()

    def test_init_default_preprocessors(self, preprocessor):
        """Test initialization with default preprocessors."""
        assert len(preprocessor.preprocessors) == 5
        assert isinstance(preprocessor.preprocessors[0], TensorPreprocessor)
        assert isinstance(preprocessor.preprocessors[1], ImageBytesPreprocessor)

    def test_init_custom_preprocessors(self):
        """Test initialization with custom preprocessors."""
        custom_preprocessors = [NumericPreprocessor()]
        preprocessor = GenericDataPreprocessor(custom_preprocessors)
        assert len(preprocessor.preprocessors) == 1
        assert isinstance(preprocessor.preprocessors[0], NumericPreprocessor)

    def test_preprocess_features_empty_data(self, preprocessor):
        """Test preprocessing empty data."""
        result = preprocessor.preprocess_features([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_preprocess_features_numeric(self, preprocessor):
        """Test preprocessing numeric data."""
        data = [1, 2, 3]
        result = preprocessor.preprocess_features(data)
        assert isinstance(result, np.ndarray)

    def test_preprocess_features_unsupported_type(self, preprocessor):
        """Test preprocessing unsupported data type."""
        # Create a custom object that no preprocessor can handle
        class UnsupportedType:
            pass
        
        data = [UnsupportedType()]
        with pytest.raises(ValueError, match="Cannot preprocess data of type"):
            preprocessor.preprocess_features(data)

    def test_add_preprocessor_at_end(self, preprocessor):
        """Test adding preprocessor at end."""
        initial_count = len(preprocessor.preprocessors)
        new_preprocessor = NumericPreprocessor()
        preprocessor.add_preprocessor(new_preprocessor)
        
        assert len(preprocessor.preprocessors) == initial_count + 1
        assert preprocessor.preprocessors[-1] is new_preprocessor

    def test_add_preprocessor_at_beginning(self, preprocessor):
        """Test adding preprocessor at beginning."""
        initial_count = len(preprocessor.preprocessors)
        new_preprocessor = NumericPreprocessor()
        preprocessor.add_preprocessor(new_preprocessor, priority=0)
        
        assert len(preprocessor.preprocessors) == initial_count + 1
        assert preprocessor.preprocessors[0] is new_preprocessor

    def test_preprocess_image_from_dict_pil(self):
        """Test _preprocess_image_from_dict with PIL image."""
        img = Image.new('RGB', (5, 5), color='red')
        result = GenericDataPreprocessor._preprocess_image_from_dict(img)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_preprocess_image_from_dict_bytes(self):
        """Test _preprocess_image_from_dict with image bytes."""
        img = Image.new('RGB', (5, 5), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        
        result = GenericDataPreprocessor._preprocess_image_from_dict(img_bytes.getvalue())
        assert isinstance(result, np.ndarray)

    def test_preprocess_image_from_dict_array(self):
        """Test _preprocess_image_from_dict with array."""
        data = [1, 2, 3]
        result = GenericDataPreprocessor._preprocess_image_from_dict(data)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=np.float32))


class TestFactoryFunctions:
    """Test factory functions for creating preprocessors."""

    def test_create_image_preprocessor_default(self):
        """Test create_image_preprocessor with default parameters."""
        preprocessor = create_image_preprocessor()
        assert isinstance(preprocessor, GenericDataPreprocessor)
        assert len(preprocessor.preprocessors) == 5

    def test_create_image_preprocessor_grayscale(self):
        """Test create_image_preprocessor with grayscale."""
        preprocessor = create_image_preprocessor(grayscale=True)
        assert isinstance(preprocessor, GenericDataPreprocessor)
        # Check that target_mode is set to 'L' for grayscale
        image_byte_proc = preprocessor.preprocessors[1]
        assert image_byte_proc.target_mode == 'L'

    def test_create_text_preprocessor(self):
        """Test create_text_preprocessor."""
        preprocessor = create_text_preprocessor()
        assert isinstance(preprocessor, GenericDataPreprocessor)
        assert len(preprocessor.preprocessors) == 3

    def test_create_tabular_preprocessor(self):
        """Test create_tabular_preprocessor."""
        preprocessor = create_tabular_preprocessor()
        assert isinstance(preprocessor, GenericDataPreprocessor)
        assert len(preprocessor.preprocessors) == 3


class TestHelperFunctions:
    """Test helper functions."""

    def test_process_dict_image_pil(self):
        """Test _process_dict_image with PIL image."""
        img = Image.new('RGB', (5, 5), color='red')
        result = _process_dict_image(img, None, True)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_process_dict_image_with_mode(self):
        """Test _process_dict_image with mode conversion."""
        img = Image.new('RGB', (5, 5), color='red')
        result = _process_dict_image(img, 'L', True)
        
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2  # Grayscale

    def test_process_image_bytes(self):
        """Test _process_image_bytes."""
        img = Image.new('RGB', (10, 10), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        
        result = _process_image_bytes(img_bytes.getvalue(), None, True, None)
        assert isinstance(result, np.ndarray)

    def test_process_image_bytes_with_resize(self):
        """Test _process_image_bytes with resize."""
        img = Image.new('RGB', (10, 10), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        
        result = _process_image_bytes(img_bytes.getvalue(), None, True, (5, 5))
        assert result.shape[:2] == (5, 5)

    def test_process_image_bytes_invalid(self):
        """Test _process_image_bytes with invalid bytes."""
        with pytest.raises(ValueError, match="Failed to process image bytes"):
            _process_image_bytes(b"invalid", None, True, None)

    def test_encode_text(self):
        """Test _encode_text."""
        result = _encode_text("hello")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert len(result) == 5  # Length of "hello"