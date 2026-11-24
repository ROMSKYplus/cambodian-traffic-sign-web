"""
Unit tests for the image preprocessing pipeline.
"""

import pytest
import numpy as np
import torch
from PIL import Image

from src.model import load_pca
from src.preprocessing import read_image_from_bytes, preprocess_image_pipeline


@pytest.fixture
def sample_pca():
    return load_pca("models/pca_ann.pkl")


def test_read_image_from_valid_bytes():
    # Create a synthetic 64x64 RGB image
    synthetic_img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    import io
    buf = io.BytesIO()
    synthetic_img.save(buf, format="PNG")
    bytes_data = buf.getvalue()

    rgb, bgr = read_image_from_bytes(bytes_data)
    assert rgb.shape == (64, 64, 3)
    assert bgr.shape == (64, 64, 3)
    assert np.allclose(rgb[:, :, 0], 128)
    assert np.allclose(rgb[:, :, 1], 64)
    assert np.allclose(rgb[:, :, 2], 32)


def test_read_image_invalid_bytes():
    with pytest.raises(ValueError, match="Unable to decode|empty"):
        read_image_from_bytes(b"corrupted_invalid_image_bytes")


def test_read_image_empty_bytes():
    with pytest.raises(ValueError, match="empty"):
        read_image_from_bytes(b"")


def test_preprocessing_tensor_shape_and_dtype(sample_pca):
    # Create synthetic test image
    img = Image.new("RGB", (128, 128), color=(200, 200, 200))
    tensor, stages = preprocess_image_pipeline(img, sample_pca)

    # 1. Check final tensor shape and type
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 50)
    assert tensor.dtype == torch.float32

    # 2. Check intermediate stage shapes
    assert stages["resized_rgb"].shape == (32, 32, 3)
    assert stages["resized_gray"].shape == (32, 32)
    assert stages["normalized_gray"].shape == (32, 32)
    assert stages["normalized_gray"].min() >= 0.0
    assert stages["normalized_gray"].max() <= 1.0
    assert stages["flattened_1024"].shape == (1, 1024)
    assert stages["pca_features_50"].shape == (1, 50)


def test_preprocessing_with_numpy_array(sample_pca):
    dummy_arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    tensor, _ = preprocess_image_pipeline(dummy_arr, sample_pca)
    assert tensor.shape == (1, 50)

