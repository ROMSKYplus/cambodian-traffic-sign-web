"""
Image preprocessing pipeline for Cambodian Traffic Sign Recognition.

Preprocessing stages:
    1. Read uploaded image (OpenCV/PIL compatible).
    2. Resize to 32 x 32 pixels.
    3. Convert to grayscale.
    4. Normalize pixel values by dividing by 255.0 (range [0, 1]).
    5. Flatten to a 1,024-dimensional feature vector.
    6. Ensure float32 data type.
    7. Apply the saved PCA transformer (1,024 -> 50 principal components).
    8. Convert to PyTorch float32 tensor of shape (1, 50).
"""

from typing import Dict, Any, Union, Tuple
import io
import numpy as np
import cv2
from PIL import Image
import torch
from sklearn.decomposition import PCA


def read_image_from_bytes(
    file_bytes: bytes
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decodes image bytes into RGB and BGR numpy arrays using OpenCV and PIL fallback.
    
    Args:
        file_bytes (bytes): Raw bytes of the uploaded image file.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (image_rgb, image_bgr).
        
    Raises:
        ValueError: If bytes cannot be decoded as a valid image.
    """
    if not file_bytes:
        raise ValueError("Uploaded file buffer is empty.")

    # Convert bytes to numpy uint8 buffer
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        # Fallback to PIL in case of rare web image encoding issues
        try:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(
                "Unable to decode uploaded image. Please ensure you upload a valid JPG, JPEG, or PNG."
            ) from e
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_rgb, img_bgr


def preprocess_image_pipeline(
    image_input: Union[bytes, np.ndarray, Image.Image],
    pca: PCA
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Full 8-stage image preprocessing pipeline matching training procedure.
    
    Args:
        image_input: Image bytes, numpy array, or PIL Image.
        pca: Pre-fitted PCA transformer loaded from models/pca_ann.pkl.
        
    Returns:
        Tuple[torch.Tensor, Dict[str, Any]]:
            - PyTorch input tensor with shape (1, 50)
            - Stages dictionary containing intermediate outputs for visualization
    """
    # Stage 1: Read image
    if isinstance(image_input, bytes):
        img_rgb, img_bgr = read_image_from_bytes(image_input)
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            # Grayscale provided
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            # RGBA provided
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif image_input.shape[2] == 3:
            img_rgb = image_input.copy()
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unexpected image array shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Stage 2: Resize to 32x32
    # cv2.resize takes (width, height)
    resized_rgb = cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_AREA)

    # Stage 3: Convert to Grayscale
    gray_32 = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)

    # Stage 4: Normalize to [0.0, 1.0]
    normalized_gray = gray_32.astype(np.float32) / 255.0

    # Stage 5 & 6: Flatten to 1,024 features of float32
    flattened_1024 = normalized_gray.reshape(1, 1024).astype(np.float32)

    # Stage 7: Apply PCA transform (1024 -> 50)
    # Do NOT fit; only transform
    pca_features = pca.transform(flattened_1024)
    pca_features = pca_features.astype(np.float32)

    # Stage 8: Convert to PyTorch Tensor of shape (1, 50)
    input_tensor = torch.from_numpy(pca_features).float()

    stages = {
        "original_rgb": img_rgb,
        "original_shape": img_rgb.shape,
        "resized_rgb": resized_rgb,
        "resized_gray": gray_32,
        "normalized_gray": normalized_gray,
        "flattened_1024": flattened_1024,
        "pca_features_50": pca_features,
        "input_tensor": input_tensor,
    }

    return input_tensor, stages

