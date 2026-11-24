"""
Unit tests for model loading, inference, and classification.
"""

from pathlib import Path
import pytest
import torch
from PIL import Image

from src.model import load_model, load_pca, TrafficSignANN
from src.prediction import (
    predict_traffic_sign,
    load_class_catalog,
    DEFAULT_INTERNAL_TO_ORIGINAL,
)


def test_model_loading_success():
    model = load_model("models/best_ann.pth")
    assert isinstance(model, TrafficSignANN)
    assert not model.training  # Must be in eval mode


def test_missing_model_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_model("models/non_existent_model.pth")


def test_missing_pca_file():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_pca("models/non_existent_pca.pkl")


def test_forward_pass_shape():
    model = load_model("models/best_ann.pth")
    dummy_input = torch.randn(1, 50)
    with torch.inference_mode():
        logits = model(dummy_input)
    assert logits.shape == (1, 10)


def test_batch_forward_pass_shape():
    model = load_model("models/best_ann.pth")
    dummy_input = torch.randn(4, 50)
    with torch.inference_mode():
        logits = model(dummy_input)
    assert logits.shape == (4, 10)


def test_class_catalog_mapping():
    catalog = load_class_catalog("config/class_names.json")
    assert len(catalog) == 10
    
    # Verify expected mapping:
    # 0 -> 0 (PedestrianCrossing)
    # 1 -> 1 (NoLeftTurn)
    # 2 -> 2 (NoRightTurn)
    # 3 -> 4 (NoEntry)
    # 4 -> 5 (SpeedLimit20km_h)
    # 5 -> 7 (NoParking)
    # 6 -> 8 (NoStopping)
    # 7 -> 9 (NoOvertaking)
    # 8 -> 22 (Stop)
    # 9 -> 33 (ChildrenCrossing)
    expected_orig_ids = [0, 1, 2, 4, 5, 7, 8, 9, 22, 33]
    for iid, orig_id in enumerate(expected_orig_ids):
        assert catalog[iid]["internal_id"] == iid
        assert catalog[iid]["original_id"] == orig_id
        assert len(catalog[iid]["name"]) > 0


def test_prediction_end_to_end_on_real_sample():
    model = load_model("models/best_ann.pth")
    pca = load_pca("models/pca_ann.pkl")
    catalog = load_class_catalog("config/class_names.json")

    sample_stop = Path("assets/sample_images/crop_class_22_Stop.jpg")
    if sample_stop.is_file():
        img = Image.open(sample_stop)
        result = predict_traffic_sign(img, model, pca, catalog, top_k=3)

        assert result["predicted_internal_id"] == 8
        assert result["predicted_original_id"] == 22
        assert "Stop" in result["predicted_name"]
        assert result["confidence"] > 0.80
        assert len(result["top_candidates"]) == 3
        # Probabilities sum to ~1.0
        assert pytest.approx(sum(result["all_probabilities"]), 1e-4) == 1.0
        assert result["inference_time_ms"] > 0.0


def test_prediction_top_k_ordering():
    model = load_model("models/best_ann.pth")
    pca = load_pca("models/pca_ann.pkl")
    catalog = load_class_catalog("config/class_names.json")

    dummy_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    result = predict_traffic_sign(dummy_img, model, pca, catalog, top_k=3)

    cands = result["top_candidates"]
    assert len(cands) == 3
    assert cands[0]["probability"] >= cands[1]["probability"] >= cands[2]["probability"]



def test_prediction_with_cropped_image():
    model = load_model("models/best_ann.pth")
    pca = load_pca("models/pca_ann.pkl")
    catalog = load_class_catalog("config/class_names.json")

    # Simulate user cropping an image
    full_img = Image.new("RGB", (200, 200), color=(50, 50, 50))
    # Place a synthetic patch
    cropped_patch = full_img.crop((20, 20, 100, 100))
    result = predict_traffic_sign(cropped_patch, model, pca, catalog, top_k=3)

    assert "predicted_name" in result
    assert "confidence_pct" in result
    assert len(result["top_candidates"]) == 3
    assert result["stages"]["resized_rgb"].shape == (32, 32, 3)
