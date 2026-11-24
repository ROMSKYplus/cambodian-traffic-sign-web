"""
Inference and prediction pipeline for Cambodian Traffic Sign Recognition.

Orchestrates preprocessing, model execution under torch.inference_mode(),
Softmax score calculation, top-K ranking, and metadata enrichment.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA

from src.preprocessing import preprocess_image_pipeline


# Default 10-class mapping in case config JSON is not loaded
DEFAULT_INTERNAL_TO_ORIGINAL = {
    0: 0,
    1: 1,
    2: 2,
    3: 4,
    4: 5,
    5: 7,
    6: 8,
    7: 9,
    8: 22,
    9: 33,
}

DEFAULT_NAMES = {
    0: "Pedestrian Crossing",
    1: "No Left Turn",
    2: "No Right Turn",
    3: "No Entry",
    4: "Speed Limit (20 km/h)",
    5: "No Parking",
    6: "No Stopping",
    7: "No Overtaking",
    8: "Stop",
    9: "Children Crossing",
}


def load_class_catalog(
    config_path: Union[str, Path] = "config/class_names.json"
) -> Dict[int, Dict[str, Any]]:
    """
    Loads class metadata catalog from config JSON.
    
    Returns:
        Dict[int, Dict[str, Any]]: Dictionary keyed by internal_id (0-9).
    """
    path = Path(config_path)
    catalog: Dict[int, Dict[str, Any]] = {}
    
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("classes", []):
                iid = int(item["internal_id"])
                catalog[iid] = item
            return catalog
        except Exception:
            pass  # Fall back to defaults below

    # Fallback default metadata if file is missing or unreadable
    for iid in range(10):
        orig_id = DEFAULT_INTERNAL_TO_ORIGINAL[iid]
        name = DEFAULT_NAMES[iid]
        catalog[iid] = {
            "internal_id": iid,
            "original_id": orig_id,
            "name": name,
            "raw_name": name.replace(" ", ""),
            "category": "Regulatory / Warning",
            "description": f"Cambodian traffic sign class {orig_id} ({name}).",
            "sample_image": f"assets/sample_images/{orig_id}.png",
        }
    return catalog


def predict_traffic_sign(
    image_input: Union[bytes, np.ndarray, Image.Image],
    model: nn.Module,
    pca: PCA,
    class_catalog: Optional[Dict[int, Dict[str, Any]]] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Runs end-to-end inference on an uploaded image.
    
    Steps:
        1. Preprocess image into 50-dim PCA features and tensor.
        2. Execute model in torch.inference_mode().
        3. Compute Softmax probabilities.
        4. Extract argmax prediction and top-K ranked candidates.
        5. Measure processing time.
        
    Args:
        image_input: Bytes, numpy array, or PIL image.
        model: Trained TrafficSignANN in eval mode.
        pca: Trained PCA transformer.
        class_catalog: Optional dictionary mapping internal_id -> metadata.
        top_k: Number of candidate classes to return (default: 3).
        
    Returns:
        Dict[str, Any]: Comprehensive prediction response.
    """
    if class_catalog is None:
        class_catalog = load_class_catalog()

    start_time = time.perf_counter()

    # Step 1-8: Preprocessing & feature extraction
    input_tensor, stages = preprocess_image_pipeline(image_input, pca)

    # Validate input dimensions
    if input_tensor.shape != (1, 50):
        raise ValueError(
            f"Expected input tensor shape (1, 50), but got {tuple(input_tensor.shape)}"
        )

    # Step 9: Model forward pass in inference mode
    raw_internals: Dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        if hasattr(model, "forward_with_internals"):
            logits_tensor, raw_internals = model.forward_with_internals(input_tensor)
        else:
            logits_tensor = model(input_tensor)

        # Validate output dimensions
        if logits_tensor.shape != (1, 10):
            raise ValueError(
                f"Expected output logits shape (1, 10), but got {tuple(logits_tensor.shape)}"
            )

        # Step 10: Apply Softmax
        probs_tensor = torch.softmax(logits_tensor, dim=1)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    logits = logits_tensor.cpu().numpy()[0]
    probabilities = probs_tensor.cpu().numpy()[0]

    # Package model internals for X-Ray visualization
    h1_acts = raw_internals["h1_post"][0].cpu().numpy() if "h1_post" in raw_internals else np.array([])
    h2_acts = raw_internals["h2_post"][0].cpu().numpy() if "h2_post" in raw_internals else np.array([])
    
    internals = {
        "pca_features": stages["pca_features_50"][0].tolist(),
        "h1_activations": [float(x) for x in h1_acts],
        "h1_active_count": int(np.sum(h1_acts > 0)),
        "h1_sparsity_pct": float(np.mean(h1_acts == 0) * 100.0) if len(h1_acts) > 0 else 0.0,
        "h2_activations": [float(x) for x in h2_acts],
        "h2_active_count": int(np.sum(h2_acts > 0)),
        "h2_sparsity_pct": float(np.mean(h2_acts == 0) * 100.0) if len(h2_acts) > 0 else 0.0,
        "logits": [float(l) for l in logits],
    }

    # Step 11: Argmax predicted class
    pred_internal_id = int(np.argmax(probabilities))
    top_class_info = class_catalog.get(pred_internal_id, {})

    # Top-K candidates
    ranked_indices = np.argsort(probabilities)[::-1][:top_k]
    top_candidates: List[Dict[str, Any]] = []

    for rank, idx in enumerate(ranked_indices, start=1):
        idx = int(idx)
        info = class_catalog.get(idx, {})
        score = float(probabilities[idx])
        top_candidates.append({
            "rank": rank,
            "internal_id": idx,
            "original_id": info.get("original_id", DEFAULT_INTERNAL_TO_ORIGINAL.get(idx, idx)),
            "name": info.get("name", f"Class {idx}"),
            "raw_name": info.get("raw_name", ""),
            "category": info.get("category", "General"),
            "description": info.get("description", ""),
            "sample_image": info.get("sample_image", ""),
            "probability": score,
            "confidence_pct": score * 100.0,
            "logit": float(logits[idx]),
        })

    return {
        "predicted_internal_id": pred_internal_id,
        "predicted_original_id": top_class_info.get("original_id", DEFAULT_INTERNAL_TO_ORIGINAL[pred_internal_id]),
        "predicted_name": top_class_info.get("name", f"Class {pred_internal_id}"),
        "category": top_class_info.get("category", "General"),
        "description": top_class_info.get("description", ""),
        "sample_image": top_class_info.get("sample_image", ""),
        "confidence": float(probabilities[pred_internal_id]),
        "confidence_pct": float(probabilities[pred_internal_id]) * 100.0,
        "inference_time_ms": elapsed_ms,
        "top_candidates": top_candidates,
        "all_probabilities": [float(p) for p in probabilities],
        "all_logits": [float(l) for l in logits],
        "stages": stages,
        "internals": internals,
    }

