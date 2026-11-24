"""
TrafficSign AI - Cambodian Traffic Sign Recognition Web Application
Design Style: Luxury Minimalist AI SaaS (Kiwi Style)
Theme: Crisp white background, black typography, vibrant lime-green accents (#4ADE80 / #22C55E),
       rounded pill buttons (border-radius: 9999px), and modern bento grid cards.
"""

import sys
from pathlib import Path
import time
import io
from typing import Optional, Dict, Any, List

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance

# Ensure src module is importable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
    import cv2
    from streamlit_cropper import st_cropper
    from src.model import load_model, load_pca, TrafficSignANN
    from src.preprocessing import preprocess_image_pipeline, read_image_from_bytes
    from src.prediction import predict_traffic_sign, load_class_catalog
except ModuleNotFoundError as e:
    st.set_page_config(page_title="Environment Setup Required — TrafficSign AI", page_icon="⚠️")
    st.error(
        f"### ⚠️ Virtual Environment Not Active\n\n"
        f"Missing package: `{e.name}`\n\n"
        "You ran Streamlit using your base Anaconda/system Python instead of the project's virtual environment where PyTorch, OpenCV, and other ML packages are installed.\n\n"
        "**To run the app correctly, stop this terminal (`Ctrl + C`) and run:**\n"
        "```bash\n"
        "source .venv/bin/activate\n"
        "streamlit run app.py\n"
        "```\n"
        "*Or run directly without activating:*\n"
        "```bash\n"
        ".venv/bin/streamlit run app.py\n"
        "```"
    )
    st.stop()


# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrafficSign AI — Cambodian Traffic Sign Recognition",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# CSS Styling Loader & HTML Helper
# -----------------------------------------------------------------------------
def load_custom_css():
    css_path = ROOT_DIR / "assets" / "styles.css"
    if css_path.is_file():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_html(html_str: str):
    """
    Renders HTML safely in Streamlit by stripping all leading whitespace from every line.
    This permanently prevents Markdown from accidentally parsing 4+ space indentations as code blocks.
    """
    cleaned = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


load_custom_css()


# -----------------------------------------------------------------------------
# Model & Asset Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_models():
    """
    Caches model and PCA transformer in memory.
    Ensures model weights and PCA components are not reloaded on every interaction.
    """
    model_path = ROOT_DIR / "models" / "best_ann.pth"
    pca_path = ROOT_DIR / "models" / "pca_ann.pkl"

    model_err = None
    pca_err = None
    model = None
    pca = None

    try:
        model = load_model(model_path, device="cpu")
    except Exception as e:
        model_err = str(e)

    try:
        pca = load_pca(pca_path)
    except Exception as e:
        pca_err = str(e)

    return model, pca, model_err, pca_err


@st.cache_data(show_spinner=False)
def get_cached_class_catalog():
    config_path = ROOT_DIR / "config" / "class_names.json"
    return load_class_catalog(config_path)


# Session state initialization
if "selected_sample_path" not in st.session_state:
    st.session_state.selected_sample_path = None
if "current_image_bytes" not in st.session_state:
    st.session_state.current_image_bytes = None
if "original_image_bytes" not in st.session_state:
    st.session_state.original_image_bytes = None
if "is_cropped" not in st.session_state:
    st.session_state.is_cropped = False
if "last_uploaded_file_id" not in st.session_state:
    st.session_state.last_uploaded_file_id = None
if "image_source_name" not in st.session_state:
    st.session_state.image_source_name = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


def reset_state():
    st.session_state.selected_sample_path = None
    st.session_state.current_image_bytes = None
    st.session_state.original_image_bytes = None
    st.session_state.is_cropped = False
    st.session_state.last_uploaded_file_id = None
    st.session_state.image_source_name = None
    st.session_state.prediction_result = None


# Load cached resources
model, pca, model_error, pca_error = get_cached_models()
catalog = get_cached_class_catalog()


# -----------------------------------------------------------------------------
# Section A: Kiwi Minimalist Navbar
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="kiwi-navbar-fixed-container">
        <div class="kiwi-navbar">
            <a href="#hero" class="kiwi-brand">
                <div class="kiwi-logo-circle">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <path d="M12 8v8"></path>
                        <path d="M8 12h8"></path>
                    </svg>
                </div>
                <span class="kiwi-brand-name">TrafficSign AI</span>
            </a>
            <div class="kiwi-nav-links">
                <a href="#hero" class="kiwi-nav-link">Home</a>
                <a href="#features" class="kiwi-nav-link">Classifier</a>
                <a href="#how-it-works" class="kiwi-nav-link">How It Works</a>
                <a href="#about-model" class="kiwi-nav-link">Architecture</a>
                <a href="#benchmark" class="kiwi-nav-link">Benchmark</a>
                <a href="#classes" class="kiwi-nav-link">Classes</a>
            </div>
            <div class="kiwi-nav-actions">
                <a href="#about-model" class="pill-btn-outline">About</a>
                <a href="#features" class="pill-btn-green">Try for free</a>
            </div>
        </div>
    </div>
    <div class="kiwi-navbar-spacer"></div>
    """,
    unsafe_allow_html=True,
)


# Check for model file availability
if model_error or pca_error:
    st.markdown(
        f"""
        <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:14px; padding:1rem 1.25rem; margin-bottom:1.5rem; color:#92400E; font-size:0.86rem;">
            <strong>⚠️ Missing or Incompatible Model Artifacts:</strong><br>
            {f"• Model error: {model_error}<br>" if model_error else ""}
            {f"• PCA error: {pca_error}<br>" if pca_error else ""}
            Please verify that both <code>models/best_ann.pth</code> and <code>models/pca_ann.pkl</code> exist in the project directory.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Section B: 2-Column Hero Section
# -----------------------------------------------------------------------------
st.markdown('<div id="hero"></div>', unsafe_allow_html=True)
col_hero_left, col_hero_right = st.columns([1.1, 1], gap="large")

with col_hero_left:
    st.markdown(
        """
        <h1 class="hero-headline">
            Recognize, Classify, and Inspect with AI Precision
        </h1>
        <p class="hero-subtitle">
            Harness the power of unsupervised PCA and deep PyTorch Neural Networks, 
            engineered specifically for high-accuracy Cambodian road traffic sign recognition.
        </p>
        
        <div class="kiwi-prompt-pill" style="max-width: 440px; margin-bottom: 2rem;">
            <span class="sparkle-badge">✨</span>
            <span>Upload or pick a test sign below to classify in real time...</span>
        </div>
        
        <div class="hero-stats-row">
            <div class="hero-stat-item">
                <div class="hero-stat-num">96.63%</div>
                <div class="hero-stat-label">Held-out test accuracy</div>
                <div class="hero-stars">★★★★★</div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-num">10</div>
                <div class="hero-stat-label">Supported sign classes</div>
                <div style="font-size:0.75rem; color:#9CA3AF; margin-top:0.35rem;">Regulatory & Warning</div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-num">95.12%</div>
                <div class="hero-stat-label">Feature reduction</div>
                <div style="font-size:0.75rem; color:#9CA3AF; margin-top:0.35rem;">1,024 → 50 PCA inputs</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_hero_right:
    hero_svg_path = ROOT_DIR / "assets" / "hero_illustration.svg"
    if hero_svg_path.is_file():
        st.image(str(hero_svg_path), use_container_width=True)


# -----------------------------------------------------------------------------
# Section C: Partner / Technology Stack Monochrome Bar
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="partner-bar">
        <div class="partner-bar-title">3,250+ Evaluated Sign Crops & Frameworks</div>
        <div class="partner-bar-logos">
            <span class="partner-logo-item">🔥 PyTorch</span>
            <span class="partner-logo-item">⚙️ Scikit-Learn</span>
            <span class="partner-logo-item">👁️ OpenCV</span>
            <span class="partner-logo-item">🎈 Streamlit</span>
            <span class="partner-logo-item">🔢 NumPy</span>
            <span class="partner-logo-item">📊 Pandas</span>
            <span class="partner-logo-item">📉 PCA (50 Comp)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Section D: "Through Our Features" Header
# -----------------------------------------------------------------------------
st.markdown('<div id="features"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-heading-row">
        <div>
            <div class="section-main-title">Through Our Features</div>
            <div class="section-main-desc">
                Unveil the powerful tools and functionalities that power our PCA + ANN model, 
                designed to classify tightly cropped road signs with maximum precision.
            </div>
        </div>
        <div>
            <a href="#benchmark" class="pill-btn-outline">Benchmarks</a>
            <a href="#classes" class="pill-btn-green">Browse Classes</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Section D: Crop & Focus Modal Dialog (Pop-up)
# -----------------------------------------------------------------------------
@st.dialog("✂️ Crop & Edit Sign", width="medium")
def crop_modal_dialog():
    source_bytes = st.session_state.get("original_image_bytes") or st.session_state.get("current_image_bytes")
    if not source_bytes:
        st.warning("No image currently loaded.")
        if st.button("Close"):
            st.rerun()
        return

    try:
        source_pil = Image.open(io.BytesIO(source_bytes))
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        return

    # Compact Toolbar: Orientation + Aspect Ratio + Contrast
    tool_col1, tool_col2, tool_col3 = st.columns([1.3, 1.2, 1], gap="small")
    with tool_col1:
        aspect_mode = st.radio(
            "Aspect Ratio:",
            ["1:1 Square (Sign)", "Freeform"],
            horizontal=True,
            key="dialog_crop_aspect",
        )
    target_aspect = (1, 1) if "1:1" in aspect_mode else None

    with tool_col2:
        rot_choice = st.radio(
            "Rotate:",
            ["0°", "90° ↻", "180°", "270° ↺"],
            horizontal=True,
            key="dialog_crop_rot",
        )

    with tool_col3:
        st.write("")
        auto_enhance = st.toggle("✨ Enhance", value=False, key="dialog_auto_enhance")

    # Apply transformations to source image
    img_transformed = source_pil.copy()
    if rot_choice == "90° ↻":
        img_transformed = img_transformed.rotate(-90, expand=True)
    elif rot_choice == "180°":
        img_transformed = img_transformed.rotate(180, expand=True)
    elif rot_choice == "270° ↺":
        img_transformed = img_transformed.rotate(90, expand=True)

    if auto_enhance:
        img_transformed = ImageOps.autocontrast(img_transformed, cutoff=2)

    orig_w, orig_h = img_transformed.size

    # Fit image to the cropper column (max 360px) so the ENTIRE photo is visible
    max_disp = 360.0
    display_scale = min(max_disp / orig_w, max_disp / orig_h)
    if max(orig_w, orig_h) < 220:
        display_scale = 220.0 / max(orig_w, orig_h)

    disp_w = max(1, int(round(orig_w * display_scale)))
    disp_h = max(1, int(round(orig_h * display_scale)))
    cropper_display_img = img_transformed.resize((disp_w, disp_h), Image.Resampling.BILINEAR)

    # Main editor row: Cropper on Left, Compact Previews on Right
    crop_c1, crop_c2 = st.columns([1.4, 1], gap="medium")
    with crop_c1:
        st.caption("Drag bounding box around sign:")
        cropper_box = st_cropper(
            cropper_display_img,
            realtime_update=True,
            box_color="#22C55E",
            aspect_ratio=target_aspect,
            return_type="box",
            key=f"modal_crop_box_{rot_choice}_{auto_enhance}",
            stroke_width=2,
            should_resize_image=False,
        )

    # Map box back to original high-res image
    full_res_crop = img_transformed
    if (
        cropper_box
        and isinstance(cropper_box, dict)
        and cropper_box.get("width", 0) > 0
        and cropper_box.get("height", 0) > 0
    ):
        b_left = max(0, min(orig_w - 1, int(round(cropper_box["left"] / display_scale))))
        b_top = max(0, min(orig_h - 1, int(round(cropper_box["top"] / display_scale))))
        b_w = max(1, int(round(cropper_box["width"] / display_scale)))
        b_h = max(1, int(round(cropper_box["height"] / display_scale)))
        b_right = max(b_left + 1, min(orig_w, b_left + b_w))
        b_bottom = max(b_top + 1, min(orig_h, b_top + b_h))
        full_res_crop = img_transformed.crop((b_left, b_top, b_right, b_bottom))

    with crop_c2:
        st.caption("Live Crop Output:")
        c_sub1, c_sub2 = st.columns([1, 1], gap="small")
        with c_sub1:
            st.image(
                full_res_crop,
                caption=f"{full_res_crop.size[0]}×{full_res_crop.size[1]}px",
                width=100,
            )
        with c_sub2:
            gray_32 = full_res_crop.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
            st.image(
                gray_32,
                caption="32×32 Gray",
                width=54,
            )

        st.markdown(
            """
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px; padding:0.45rem 0.65rem; font-size:0.74rem; color:#166534; margin-top:0.35rem; line-height:1.3;">
                <strong>Pipeline Ready:</strong><br>
                50-dim PCA projection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    btn_c1, btn_c2, btn_c3 = st.columns([1.1, 1.4, 0.8], gap="small")
    with btn_c1:
        if st.button("✅ Apply Crop", type="primary", use_container_width=True, key="btn_apply_crop_modal"):
            buf = io.BytesIO()
            full_res_crop.convert("RGB").save(buf, format="JPEG", quality=95)
            if not st.session_state.get("original_image_bytes"):
                st.session_state.original_image_bytes = source_bytes
            st.session_state.current_image_bytes = buf.getvalue()
            st.session_state.is_cropped = True
            st.session_state.prediction_result = None
            st.rerun()

    with btn_c2:
        if st.button("🚀 Apply & Predict", use_container_width=True, key="btn_apply_predict_modal"):
            buf = io.BytesIO()
            full_res_crop.convert("RGB").save(buf, format="JPEG", quality=95)
            if not st.session_state.get("original_image_bytes"):
                st.session_state.original_image_bytes = source_bytes
            st.session_state.current_image_bytes = buf.getvalue()
            st.session_state.is_cropped = True
            if model is not None and pca is not None:
                res = predict_traffic_sign(
                    image_input=st.session_state.current_image_bytes,
                    model=model,
                    pca=pca,
                    class_catalog=catalog,
                    top_k=3,
                )
                st.session_state.prediction_result = res
            st.rerun()

    with btn_c3:
        if st.button("Cancel", use_container_width=True, key="btn_cancel_crop_modal"):
            st.rerun()


# -----------------------------------------------------------------------------
# Section E: Interactive Classifier (Bento Grid Row 1)
# -----------------------------------------------------------------------------
bento_col1, bento_col2 = st.columns([1.1, 1], gap="medium")

with bento_col1:
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-title">Instant Sign Classification</div>
            <div class="card-desc">Upload a tightly cropped Cambodian traffic sign image in JPG, JPEG, or PNG format.</div>
            <div class="kiwi-prompt-pill">
                <span class="sparkle-badge">✨</span>
                <span>Select a cropped sign image to feed into the 50-dim PCA + ANN pipeline...</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a cropped image containing a single Cambodian traffic sign.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            # Only trigger reset if a brand new file was selected in the uploader
            if st.session_state.get("last_uploaded_file_id") != file_id:
                st.session_state.last_uploaded_file_id = file_id
                st.session_state.current_image_bytes = file_bytes
                st.session_state.original_image_bytes = file_bytes
                st.session_state.is_cropped = False
                st.session_state.image_source_name = uploaded_file.name
                st.session_state.prediction_result = None

        # Display preview & controls inside the card
        if st.session_state.current_image_bytes:
            st.divider()
            try:
                pil_raw = Image.open(io.BytesIO(st.session_state.current_image_bytes))
            except Exception:
                st.error("Failed to decode uploaded image.")
                pil_raw = None

            if pil_raw is not None:
                sub_c1, sub_c2 = st.columns([1, 1.8], gap="small")
                with sub_c1:
                    st.image(
                        pil_raw,
                        caption=f"{pil_raw.size[0]}×{pil_raw.size[1]}px",
                        use_container_width=True,
                    )
                with sub_c2:
                    st.caption(f"**Loaded:** `{st.session_state.image_source_name}`")
                    if st.session_state.get("is_cropped", False):
                        st.markdown(
                            """
                            <div style="display:inline-flex; align-items:center; gap:0.35rem; background:#DCFCE7; color:#15803D; font-size:0.75rem; font-weight:700; padding:0.25rem 0.65rem; border-radius:9999px; margin-bottom:0.4rem;">
                                <span>✂️ Cropped Sign Applied</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button("↩️ Reset to Original", key="btn_revert_crop", use_container_width=True):
                            if st.session_state.get("original_image_bytes"):
                                st.session_state.current_image_bytes = st.session_state.original_image_bytes
                                st.session_state.is_cropped = False
                                st.session_state.prediction_result = None
                                st.rerun()
                    else:
                        st.caption("💡 *Need to isolate the sign? Click **Edit / Crop** below to pop up the cropper.*")

                # Action controls row: Predict, Edit / Crop, Clear
                ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.3, 1.2, 0.9])
                with ctrl_c1:
                    btn_run = st.button("🚀 Predict Sign", type="primary", use_container_width=True, key="btn_run_predict")
                with ctrl_c2:
                    if st.button("✂️ Edit / Crop", use_container_width=True, key="btn_open_crop_modal"):
                        crop_modal_dialog()
                with ctrl_c3:
                    if st.button("🔄 Clear", use_container_width=True, key="btn_clear_upload"):
                        reset_state()
                        st.rerun()

                if btn_run and model is not None and pca is not None:
                    with st.spinner("Processing with PCA + ANN..."):
                        try:
                            res = predict_traffic_sign(
                                image_input=st.session_state.current_image_bytes,
                                model=model,
                                pca=pca,
                                class_catalog=catalog,
                                top_k=3,
                            )
                            st.session_state.prediction_result = res
                            st.rerun()
                        except Exception as e:
                            st.error(f"Inference error: {e}")

with bento_col2:
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-title">Dataset Benchmark Quick-Picker</div>
            <div class="card-desc">Click any verified ground-truth dataset crop to immediately evaluate the model:</div>
            """,
            unsafe_allow_html=True,
        )

        sample_files = [
            ("0.png", "assets/sample_images/crop_class_0_PedestrianCrossing.jpg", "Pedestrian"),
            ("1.png", "assets/sample_images/crop_class_1_NoLeftTurn.jpg", "No Left"),
            ("2.png", "assets/sample_images/crop_class_2_NoRightTurn.jpg", "No Right"),
            ("4.png", "assets/sample_images/crop_class_4_NoEntry.jpg", "No Entry"),
            ("5.png", "assets/sample_images/crop_class_5_SpeedLimit20.jpg", "Speed 20"),
            ("7.png", "assets/sample_images/crop_class_7_NoParking.jpg", "No Parking"),
            ("8.png", "assets/sample_images/crop_class_8_NoStopping.jpg", "No Stop"),
            ("9.png", "assets/sample_images/crop_class_9_NoOvertaking.jpg", "No Pass"),
            ("22.png", "assets/sample_images/crop_class_22_Stop.jpg", "Stop"),
            ("33.png", "assets/sample_images/crop_class_33_ChildrenCrossing.jpg", "Children"),
        ]

        sample_cols = st.columns(5)
        for i, (icon_name, crop_path, short_label) in enumerate(sample_files):
            with sample_cols[i % 5]:
                resolved_crop = ROOT_DIR / crop_path
                resolved_icon = ROOT_DIR / "assets" / "sample_images" / icon_name
                target_img = resolved_crop if resolved_crop.is_file() else resolved_icon

                if target_img.is_file():
                    st.image(str(target_img), use_container_width=True)
                    if st.button(short_label, key=f"btn_sample_kiwi_{i}", use_container_width=True):
                        with open(target_img, "rb") as f:
                            sample_bytes = f.read()
                        st.session_state.last_uploaded_file_id = None
                        st.session_state.current_image_bytes = sample_bytes
                        st.session_state.original_image_bytes = sample_bytes
                        st.session_state.is_cropped = False
                        st.session_state.image_source_name = target_img.name
                        # Instant prediction on sample click
                        if model is not None and pca is not None:
                            res = predict_traffic_sign(
                                image_input=st.session_state.current_image_bytes,
                                model=model,
                                pca=pca,
                                class_catalog=catalog,
                                top_k=3,
                            )
                            st.session_state.prediction_result = res
                        st.rerun()


# -----------------------------------------------------------------------------
# Section F: Prediction Results (If Available)
# -----------------------------------------------------------------------------
if st.session_state.prediction_result:
    res = st.session_state.prediction_result
    st.write("")
    res_col_left, res_col_right = st.columns([1.1, 1], gap="medium")

    with res_col_left:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="result-box-kiwi">
                    <span class="result-badge-green">Primary Detection</span>
                    <div class="result-sign-title">{res['predicted_name']}</div>
                    <p style="margin: 0.4rem 0 0.75rem 0; font-size: 0.86rem; color: #4B5563;">
                        {res.get('description', '')}
                    </p>
                    <div class="result-pills-row">
                        <span class="meta-pill highlight">Confidence: {res['confidence_pct']:.2f}%</span>
                        <span class="meta-pill">Original Class ID: <strong>{res['predicted_original_id']}</strong></span>
                        <span class="meta-pill">Model Index: <strong>{res['predicted_internal_id']}</strong></span>
                        <span class="meta-pill">Latency: <strong>{res['inference_time_ms']:.2f} ms</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "ℹ️ **Confidence Note:** Softmax confidence reflects normalized activation across the 10 trained classes, not an absolute guarantee."
            )

    with res_col_right:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">Top 3 Predicted Candidates</div>
                <div class="card-desc">Comparative Softmax score distribution across top model outputs:</div>
                """,
                unsafe_allow_html=True,
            )

            for cand in res["top_candidates"]:
                rk = cand["rank"]
                nm = cand["name"]
                orig_id = cand["original_id"]
                pct = cand["confidence_pct"]
                rk_class = f"r{rk}"

                st.markdown(
                    f"""
                    <div class="prob-row-kiwi">
                        <div class="prob-header-kiwi">
                            <span><strong>#{rk}</strong> {nm} <span style="color:#9CA3AF; font-size:0.75rem;">(ID {orig_id})</span></span>
                            <span><strong>{pct:.2f}%</strong></span>
                        </div>
                        <div class="prob-track-kiwi">
                            <div class="prob-fill-kiwi {rk_class}" style="width: {max(pct, 2.5)}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("Inspect All 10 Class Probabilities & Logits"):
                dist_list = []
                for iid, prob in enumerate(res["all_probabilities"]):
                    c_info = catalog.get(iid, {})
                    dist_list.append({
                        "Model Index": iid,
                        "Original ID": c_info.get("original_id", iid),
                        "Sign Name": c_info.get("name", f"Class {iid}"),
                        "Raw Logit": f"{res['all_logits'][iid]:.4f}",
                        "Probability": f"{prob * 100:.3f}%",
                    })
                st.dataframe(pd.DataFrame(dist_list), use_container_width=True, hide_index=True)


    # -------------------------------------------------------------------------
    # Preprocessing Pipeline Bento Card
    # -------------------------------------------------------------------------
    stages = res["stages"]
    st.write("")
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-title">Real-Time Feature Transformation Stages</div>
            <div class="card-desc">
                Mathematical image transformations: 
                <strong>1,024 raw pixels → 50 PCA components → ANN logits</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        stage_c1, stage_c2, stage_c3, stage_c4 = st.columns(4, gap="medium")

        with stage_c1:
            st.markdown(
                """
                <div class="stage-mini-box">
                    <div class="stage-mini-title">Stage 1: Input Crop</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(stages["original_rgb"], use_container_width=True)
            st.markdown(
                f"<div class='stage-mini-meta'>{stages['original_shape'][1]}×{stages['original_shape'][0]} RGB</div>",
                unsafe_allow_html=True,
            )

        with stage_c2:
            st.markdown(
                """
                <div class="stage-mini-box">
                    <div class="stage-mini-title">Stage 2: Resized 32×32</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(stages["resized_rgb"], use_container_width=True)
            st.markdown(
                "<div class='stage-mini-meta'>32×32 RGB (1,024 px)</div>",
                unsafe_allow_html=True,
            )

        with stage_c3:
            st.markdown(
                """
                <div class="stage-mini-box">
                    <div class="stage-mini-title">Stage 3: Grayscale</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(stages["resized_gray"], use_container_width=True, clamp=True)
            st.markdown(
                "<div class='stage-mini-meta'>Single Channel uint8</div>",
                unsafe_allow_html=True,
            )

        with stage_c4:
            st.markdown(
                """
                <div class="stage-mini-box">
                    <div class="stage-mini-title">Stage 4: Normalized</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(stages["normalized_gray"], use_container_width=True, clamp=True)
            st.markdown(
                f"<div class='stage-mini-meta'>[{stages['normalized_gray'].min():.2f}, {stages['normalized_gray'].max():.2f}]</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="kiwi-pipeline-banner">
                <span>📷 Raw Crop</span>
                <span>➔</span>
                <span>📐 Resized (32×32)</span>
                <span>➔</span>
                <span>⬛ Grayscale (32×32)</span>
                <span>➔</span>
                <span>📊 Vector (1,024 float32)</span>
                <span>➔</span>
                <span>📉 PCA Projection (50 components)</span>
                <span>➔</span>
                <span>🧠 ANN (100 ➔ 50 ➔ 10)</span>
                <span>➔</span>
                <span>🟢 Classification</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Section G: How the Model Works Inside (Glass-Box Introspection Engine)
# -----------------------------------------------------------------------------
st.write("")
st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)

# Determine active internals source
active_internals = None
active_sign_label = "Pedestrian Crossing"
active_confidence = 0.994
is_live_mode = False

if st.session_state.prediction_result and "internals" in st.session_state.prediction_result:
    active_internals = st.session_state.prediction_result["internals"]
    active_sign_label = st.session_state.prediction_result.get("predicted_name", "Traffic Sign")
    active_confidence = st.session_state.prediction_result.get("confidence", 0.0)
    is_live_mode = True
else:
    # Compute baseline internals using sample image
    sample_file = "assets/sample_images/crop_class_0_PedestrianCrossing.jpg"
    if Path(sample_file).is_file() and model is not None and pca is not None:
        try:
            with open(sample_file, "rb") as f:
                sample_bytes = f.read()
            sample_res = predict_traffic_sign(sample_bytes, model, pca, catalog)
            active_internals = sample_res.get("internals")
            active_sign_label = sample_res.get("predicted_name", "Pedestrian Crossing")
            active_confidence = sample_res.get("confidence", 0.994)
        except Exception:
            active_internals = None

render_html(
    """
    <div class="section-heading-row" style="margin-top: 1.5rem;">
        <div>
            <div class="section-main-title">How the Model Works Inside</div>
            <div class="section-main-desc">
                Glass-box introspection: explore unsupervised PCA feature vectors, ReLU neuron activations, layer sparsity, and decision boundaries.
            </div>
        </div>
    </div>
    """
)

if active_internals is not None:
    if is_live_mode:
        render_html(
            f"""
            <div class="glassbox-mode-pill live">
                <span class="pulse-dot"></span>
                <span><strong>Live Active Inspection:</strong> Introspecting real-time activations for <strong>{active_sign_label}</strong> ({active_confidence * 100:.1f}% confidence)</span>
            </div>
            """
        )
    else:
        render_html(
            f"""
            <div class="glassbox-mode-pill baseline">
                <span>💡</span>
                <span><strong>Baseline Demonstration:</strong> Showing layer activations for sample sign (<strong>{active_sign_label}</strong>). Upload or select any sign above to see live updates.</span>
            </div>
            """
        )

    tab_xray, tab_pipeline, tab_topology = st.tabs([
        "⚡ Live Layer Activations (Image X-Ray)",
        "🔬 4-Stage Mathematical Pipeline",
        "📐 Interactive Network Topology Diagram",
    ])

    # -------------------------------------------------------------------------
    # Tab 1: Live Layer Activations (Image X-Ray)
    # -------------------------------------------------------------------------
    with tab_xray:
        with st.container(border=True):
            pca_feats = active_internals.get("pca_features", [])
            h1_acts = active_internals.get("h1_activations", [])
            h1_active_count = active_internals.get("h1_active_count", 0)
            h1_sparsity = active_internals.get("h1_sparsity_pct", 0.0)
            h2_acts = active_internals.get("h2_activations", [])
            h2_active_count = active_internals.get("h2_active_count", 0)
            h2_sparsity = active_internals.get("h2_sparsity_pct", 0.0)
            logits_list = active_internals.get("logits", [])

            # Top metric strip
            render_html(
                f"""
                <div class="glassbox-metric-grid">
                    <div class="glassbox-metric-box">
                        <div class="glassbox-metric-label">PCA Features</div>
                        <div class="glassbox-metric-val">{len(pca_feats)}</div>
                        <div class="glassbox-metric-sub">95.1% Compression</div>
                    </div>
                    <div class="glassbox-metric-box">
                        <div class="glassbox-metric-label">Layer 1 Firing</div>
                        <div class="glassbox-metric-val">{h1_active_count} / {len(h1_acts)}</div>
                        <div class="glassbox-metric-sub">{h1_sparsity:.1f}% ReLU Sparsity</div>
                    </div>
                    <div class="glassbox-metric-box">
                        <div class="glassbox-metric-label">Layer 2 Firing</div>
                        <div class="glassbox-metric-val">{h2_active_count} / {len(h2_acts)}</div>
                        <div class="glassbox-metric-sub">{h2_sparsity:.1f}% ReLU Sparsity</div>
                    </div>
                    <div class="glassbox-metric-box">
                        <div class="glassbox-metric-label">Top Decision Logit</div>
                        <div class="glassbox-metric-val">{max(logits_list) if logits_list else 0.0:.2f}</div>
                        <div class="glassbox-metric-sub">{active_confidence * 100:.1f}% Softmax</div>
                    </div>
                </div>
                """
            )

            # Sub-module 1: PCA Components Bar Chart
            st.markdown(
                """
                <div class="glassbox-subcard-header">
                    <div>
                        <div class="glassbox-subcard-title">Stage 1: 50 Principal Components Spectrum (z₁ ... z₅₀)</div>
                        <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.15rem;">
                            Orthogonal eigenvector projections capturing 95.12% variance of the 32×32 grayscale crop.
                        </div>
                    </div>
                    <span class="glassbox-subcard-badge">50 Dimensions</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if pca_feats:
                pca_df = pd.DataFrame({
                    "Component": [f"PC{i+1:02d}" for i in range(len(pca_feats))],
                    "Projection Value": [round(x, 4) for x in pca_feats],
                })
                st.bar_chart(pca_df, x="Component", y="Projection Value", color="#22C55E", height=230)

            render_html(
                """
                <div class="glassbox-callout">
                    <strong>💡 PCA Insight:</strong> Principal Component Analysis eliminates spatial pixel collinearity without requiring class labels. Dominant peaks correspond to prominent traffic sign edges and geometric frames, while background road and sky textures have near-zero component values.
                </div>
                """
            )

            st.write("")
            st.divider()
            st.write("")

            # Sub-module 2: Hidden Layer 1 Activations
            st.markdown(
                f"""
                <div class="glassbox-subcard-header">
                    <div>
                        <div class="glassbox-subcard-title">Stage 2: Hidden Layer 1 ReLU Activation Spectrum (100 Neurons)</div>
                        <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.15rem;">
                            Linear(50, 100) + ReLU(). Highlighting selective neuron firing and biological inhibition.
                        </div>
                    </div>
                    <span class="glassbox-subcard-badge">{h1_active_count} Active ({h1_sparsity:.1f}% Dormant)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if h1_acts:
                h1_df = pd.DataFrame({
                    "Neuron": [f"N{i+1:03d}" for i in range(len(h1_acts))],
                    "ReLU Activation": [round(max(0.0, x), 4) for x in h1_acts],
                })
                st.bar_chart(h1_df, x="Neuron", y="ReLU Activation", color="#10B981", height=230)

            render_html(
                f"""
                <div class="glassbox-callout">
                    <strong>🧠 ReLU Sparsity Principle:</strong> The activation function $\\text{{ReLU}}(x) = \\max(0, x)$ clamps negative pre-activations to exactly 0.0. In this layer, <strong>{h1_sparsity:.1f}% of neurons are dormant</strong>, creating a sparse, decoupled internal representation that resists overfitting.
                </div>
                """
            )

            st.write("")
            st.divider()
            st.write("")

            # Sub-module 3: Hidden Layer 2 Activations
            st.markdown(
                f"""
                <div class="glassbox-subcard-header">
                    <div>
                        <div class="glassbox-subcard-title">Stage 3: Hidden Layer 2 ReLU Activation Spectrum (50 Neurons)</div>
                        <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.15rem;">
                            Linear(100, 50) + ReLU(). High-level semantic traffic sign archetype feature detectors.
                        </div>
                    </div>
                    <span class="glassbox-subcard-badge">{h2_active_count} Active ({h2_sparsity:.1f}% Dormant)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if h2_acts:
                h2_df = pd.DataFrame({
                    "Neuron": [f"N{i+1:02d}" for i in range(len(h2_acts))],
                    "ReLU Activation": [round(max(0.0, x), 4) for x in h2_acts],
                })
                st.bar_chart(h2_df, x="Neuron", y="ReLU Activation", color="#059669", height=230)

            render_html(
                """
                <div class="glassbox-callout">
                    <strong>🔍 Semantic Feature Synthesis:</strong> Hidden Layer 2 recombines edge activations into high-level geometric signatures: circular prohibition borders, triangular hazard enclosures, pedestrian silhouettes, and directional arrows.
                </div>
                """
            )

            st.write("")
            st.divider()
            st.write("")

            # Sub-module 4: Output Logits vs Softmax Probabilities
            st.markdown(
                """
                <div class="glassbox-subcard-header">
                    <div>
                        <div class="glassbox-subcard-title">Stage 4: Output Layer Logits vs. Normalized Softmax Probabilities (10 Classes)</div>
                        <div style="font-size: 0.82rem; color: #6B7280; margin-top: 0.15rem;">
                            Linear(50, 10) producing unnormalized real logits, transformed into a valid probability distribution by Softmax.
                        </div>
                    </div>
                    <span class="glassbox-subcard-badge">10 Classes</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Build comparison table
            if logits_list and catalog:
                exp_logits = np.exp(np.array(logits_list) - np.max(logits_list))
                calc_probs = exp_logits / np.sum(exp_logits)
                
                rows_data = []
                for idx_c, logit_val in enumerate(logits_list):
                    c_info = catalog.get(idx_c, {})
                    c_name = c_info.get("name", f"Class {idx_c}")
                    c_orig = c_info.get("original_id", idx_c)
                    c_prob = float(calc_probs[idx_c])
                    rows_data.append({
                        "Internal ID": idx_c,
                        "Original ID": c_orig,
                        "Traffic Sign Class": c_name,
                        "Raw Logit (z)": f"{logit_val:+.3f}",
                        "Softmax Probability": f"{c_prob * 100:.2f}%",
                        "_prob_raw": c_prob,
                    })

                sorted_rows = sorted(rows_data, key=lambda r: r["_prob_raw"], reverse=True)
                for rank_idx, r in enumerate(sorted_rows, start=1):
                    r["Rank"] = f"#{rank_idx}"

                out_display_df = pd.DataFrame(sorted_rows)[[
                    "Rank", "Original ID", "Traffic Sign Class", "Raw Logit (z)", "Softmax Probability"
                ]]
                st.dataframe(out_display_df, use_container_width=True, hide_index=True)

            render_html(
                """
                <div class="glassbox-callout">
                    <strong>📐 Softmax Mechanics:</strong> Raw logits are unbounded scores in $(-\\infty, +\\infty)$. The Softmax operator $P(y=k) = \\frac{e^{z_k}}{\\sum_j e^{z_j}}$ exponentiates and normalizes each logit. Even negative logits (e.g. $-11.4$) contribute near-zero probability ($e^{-11.4} \\approx 10^{-5}$), while the dominant positive logit decisively wins the MAP decision.
                </div>
                """
            )

    # -------------------------------------------------------------------------
    # Tab 2: 4-Stage Mathematical Pipeline
    # -------------------------------------------------------------------------
    with tab_pipeline:
        with st.container(border=True):
            p_col1, p_col2 = st.columns(2, gap="medium")

            with p_col1:
                with st.container(border=True):
                    render_html(
                        """
                        <div class="math-card-step">Stage 1 · Spatial Discretization</div>
                        <div class="math-card-title">Pixel Grid Flattening & Scaling</div>
                        """
                    )
                    st.latex(r"x_{i,j} \in [0, 255] \xrightarrow{/ 255} y_{i,j} \in [0, 1] \xrightarrow{\text{Flatten}} \mathbf{x} \in \mathbb{R}^{1024}")
                    render_html(
                        """
                        <div class="math-card-desc">
                            The raw traffic sign crop is scaled to 32×32 pixels and transformed to single-channel grayscale via luminance weighting <em>Y = 0.299R + 0.587G + 0.114B</em>. Pixel intensities are min-max scaled into [0.0, 1.0] and flattened into a 1,024-dimensional feature vector.
                        </div>
                        <div class="math-card-why">
                            🎯 <strong>Problem Solved:</strong> Color channels vary wildly under tropical sunlight and rain; grayscale ensures focus on road sign geometry and pictograms.
                        </div>
                        """
                    )

            with p_col2:
                with st.container(border=True):
                    render_html(
                        """
                        <div class="math-card-step">Stage 2 · Unsupervised Projection</div>
                        <div class="math-card-title">PCA Dimensionality Reduction</div>
                        """
                    )
                    st.latex(r"\mathbf{z} = (\mathbf{x} - \boldsymbol{\mu}) \cdot \mathbf{W}_{50}, \quad \mathbf{W}_{50} \in \mathbb{R}^{1024 \times 50}")
                    render_html(
                        """
                        <div class="math-card-desc">
                            Principal Component Analysis computes the empirical covariance matrix across the training set and extracts the top 50 orthogonal eigenvectors. The 1,024-dimensional vector is projected onto this optimal variance subspace.
                        </div>
                        <div class="math-card-why">
                            🎯 <strong>Problem Solved:</strong> Eliminates spatial collinearity among neighboring pixels and slashes first-layer parameters from 102,400 down to 5,000 (95.12% compression).
                        </div>
                        """
                    )

            st.write("")
            p_col3, p_col4 = st.columns(2, gap="medium")

            with p_col3:
                with st.container(border=True):
                    render_html(
                        """
                        <div class="math-card-step">Stage 3 · Deep Feature Propagation</div>
                        <div class="math-card-title">Multilayer Non-Linear Processing</div>
                        """
                    )
                    st.latex(r"\mathbf{h}_1 = \text{ReLU}(\mathbf{z}\mathbf{W}_1 + \mathbf{b}_1), \quad \mathbf{h}_2 = \text{ReLU}(\mathbf{h}_1\mathbf{W}_2 + \mathbf{b}_2)")
                    render_html(
                        """
                        <div class="math-card-desc">
                            The 50 PCA features propagate through two hidden layers (100 neurons and 50 neurons). Each linear affine transformation is followed by the non-linear Rectified Linear Unit activation function <em>ReLU(v) = max(0, v)</em>.
                        </div>
                        <div class="math-card-why">
                            🎯 <strong>Problem Solved:</strong> Linear PCA alone cannot separate intertwined sign classes; stacked ReLUs construct complex piecewise linear decision surfaces.
                        </div>
                        """
                    )

            with p_col4:
                with st.container(border=True):
                    render_html(
                        """
                        <div class="math-card-step">Stage 4 · Statistical Decision</div>
                        <div class="math-card-title">Softmax Probability & MAP Classification</div>
                        """
                    )
                    st.latex(r"P(y=k \mid \mathbf{x}) = \frac{\exp(y_k)}{\sum_{j=1}^{10} \exp(y_j)}, \quad \hat{y} = \arg\max_k P(y=k \mid \mathbf{x})")
                    render_html(
                        """
                        <div class="math-card-desc">
                            The output layer computes 10 class logits <em>y = h₂W₃ + b₃</em>. Softmax normalizes them into a valid posterior probability distribution summing to 1.0. The final predicted class is chosen via Maximum A Posteriori (MAP).
                        </div>
                        <div class="math-card-why">
                            🎯 <strong>Problem Solved:</strong> Produces a continuous uncertainty metric and ranking across candidate signs rather than a blunt binary output.
                        </div>
                        """
                    )

    # -------------------------------------------------------------------------
    # Tab 3: Interactive Network Topology Diagram
    # -------------------------------------------------------------------------
    with tab_topology:
        with st.container(border=True):
            render_html(
                """
                <div class="topology-container">
                    <div style="font-weight: 800; font-size: 1.1rem; color: #111827; margin-bottom: 0.35rem;">
                        Feedforward ANN Computational Graph
                    </div>
                    <div style="font-size: 0.82rem; color: #6B7280; margin-bottom: 1.25rem; text-align: center;">
                        Complete layer connectivity, tensor shapes, and parameter counts from input to output logits.
                    </div>
                    <svg class="topology-svg" viewBox="0 0 860 310" xmlns="http://www.w3.org/2000/svg">
                        <!-- Background Grid lines -->
                        <line x1="120" y1="40" x2="120" y2="270" stroke="#E5E7EB" stroke-dasharray="3 3"/>
                        <line x1="330" y1="40" x2="330" y2="270" stroke="#E5E7EB" stroke-dasharray="3 3"/>
                        <line x1="540" y1="40" x2="540" y2="270" stroke="#E5E7EB" stroke-dasharray="3 3"/>
                        <line x1="750" y1="40" x2="750" y2="270" stroke="#E5E7EB" stroke-dasharray="3 3"/>
                        
                        <!-- Connecting arrows between layers -->
                        <!-- Input to Hidden 1 -->
                        <path d="M170 150 L270 150" stroke="#22C55E" stroke-width="2.5" marker-end="url(#arrow)"/>
                        <rect x="185" y="125" width="70" height="20" rx="4" fill="#DCFCE7" stroke="#86EFAC"/>
                        <text x="220" y="139" text-anchor="middle" font-size="9" font-weight="700" fill="#065F46">5,100 W+b</text>

                        <!-- Hidden 1 to Hidden 2 -->
                        <path d="M380 150 L480 150" stroke="#10B981" stroke-width="2.5" marker-end="url(#arrow)"/>
                        <rect x="395" y="125" width="70" height="20" rx="4" fill="#D1FAE5" stroke="#6EE7B7"/>
                        <text x="430" y="139" text-anchor="middle" font-size="9" font-weight="700" fill="#047857">5,050 W+b</text>

                        <!-- Hidden 2 to Output -->
                        <path d="M590 150 L690 150" stroke="#059669" stroke-width="2.5" marker-end="url(#arrow)"/>
                        <rect x="610" y="125" width="60" height="20" rx="4" fill="#ECFDF5" stroke="#A7F3D0"/>
                        <text x="640" y="139" text-anchor="middle" font-size="9" font-weight="700" fill="#065F46">510 W+b</text>

                        <defs>
                            <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                                <path d="M 0 1 L 8 5 L 0 9 z" fill="#111827" />
                            </marker>
                        </defs>

                        <!-- Layer 0: Input Box -->
                        <g transform="translate(60, 65)">
                            <rect width="110" height="170" rx="14" fill="#FFFFFF" stroke="#111827" stroke-width="2"/>
                            <rect x="12" y="12" width="86" height="24" rx="12" fill="#F3F4F6"/>
                            <text x="55" y="28" text-anchor="middle" font-size="10" font-weight="800" fill="#111827">LAYER 0</text>
                            <text x="55" y="65" text-anchor="middle" font-size="13" font-weight="800" fill="#111827">PCA Input</text>
                            <text x="55" y="85" text-anchor="middle" font-size="10" font-weight="600" fill="#6B7280">(batch, 50)</text>
                            <circle cx="55" cy="115" r="5" fill="#22C55E"/>
                            <circle cx="55" cy="130" r="5" fill="#22C55E"/>
                            <circle cx="55" cy="145" r="5" fill="#22C55E"/>
                            <text x="55" y="162" text-anchor="middle" font-size="9" font-weight="600" fill="#9CA3AF">50 Features</text>
                        </g>

                        <!-- Layer 1: Hidden 1 Box -->
                        <g transform="translate(270, 50)">
                            <rect width="110" height="200" rx="14" fill="#FFFFFF" stroke="#22C55E" stroke-width="2.5"/>
                            <rect x="12" y="12" width="86" height="24" rx="12" fill="#DCFCE7"/>
                            <text x="55" y="28" text-anchor="middle" font-size="10" font-weight="800" fill="#065F46">LAYER 1</text>
                            <text x="55" y="65" text-anchor="middle" font-size="13" font-weight="800" fill="#111827">Hidden 1</text>
                            <text x="55" y="85" text-anchor="middle" font-size="10" font-weight="600" fill="#6B7280">(batch, 100)</text>
                            <rect x="25" y="98" width="60" height="18" rx="4" fill="#F3F4F6"/>
                            <text x="55" y="111" text-anchor="middle" font-size="9" font-weight="700" fill="#111827">ReLU()</text>
                            <circle cx="45" cy="135" r="4.5" fill="#10B981"/>
                            <circle cx="65" cy="135" r="4.5" fill="#10B981"/>
                            <circle cx="45" cy="152" r="4.5" fill="#E5E7EB"/>
                            <circle cx="65" cy="152" r="4.5" fill="#10B981"/>
                            <circle cx="55" cy="168" r="4.5" fill="#E5E7EB"/>
                            <text x="55" y="188" text-anchor="middle" font-size="9" font-weight="700" fill="#059669">100 Neurons</text>
                        </g>

                        <!-- Layer 2: Hidden 2 Box -->
                        <g transform="translate(480, 65)">
                            <rect width="110" height="170" rx="14" fill="#FFFFFF" stroke="#10B981" stroke-width="2"/>
                            <rect x="12" y="12" width="86" height="24" rx="12" fill="#D1FAE5"/>
                            <text x="55" y="28" text-anchor="middle" font-size="10" font-weight="800" fill="#047857">LAYER 2</text>
                            <text x="55" y="65" text-anchor="middle" font-size="13" font-weight="800" fill="#111827">Hidden 2</text>
                            <text x="55" y="85" text-anchor="middle" font-size="10" font-weight="600" fill="#6B7280">(batch, 50)</text>
                            <rect x="25" y="98" width="60" height="18" rx="4" fill="#F3F4F6"/>
                            <text x="55" y="111" text-anchor="middle" font-size="9" font-weight="700" fill="#111827">ReLU()</text>
                            <circle cx="55" cy="132" r="5" fill="#059669"/>
                            <circle cx="55" cy="148" r="5" fill="#059669"/>
                            <text x="55" y="162" text-anchor="middle" font-size="9" font-weight="700" fill="#047857">50 Neurons</text>
                        </g>

                        <!-- Layer 3: Output Box -->
                        <g transform="translate(690, 65)">
                            <rect width="110" height="170" rx="14" fill="#FFFFFF" stroke="#111827" stroke-width="2"/>
                            <rect x="12" y="12" width="86" height="24" rx="12" fill="#FEF3C7"/>
                            <text x="55" y="28" text-anchor="middle" font-size="10" font-weight="800" fill="#92400E">OUTPUT</text>
                            <text x="55" y="65" text-anchor="middle" font-size="13" font-weight="800" fill="#111827">Class Logits</text>
                            <text x="55" y="85" text-anchor="middle" font-size="10" font-weight="600" fill="#6B7280">(batch, 10)</text>
                            <rect x="20" y="98" width="70" height="18" rx="4" fill="#FEF08A"/>
                            <text x="55" y="111" text-anchor="middle" font-size="9" font-weight="800" fill="#854D0E">Softmax()</text>
                            <circle cx="55" cy="132" r="5" fill="#F59E0B"/>
                            <circle cx="55" cy="148" r="5" fill="#F59E0B"/>
                            <text x="55" y="162" text-anchor="middle" font-size="9" font-weight="700" fill="#D97706">10 Probabilities</text>
                        </g>
                    </svg>
                </div>
                """
            )

            # Parameter breakdown table
            param_df = pd.DataFrame([
                {
                    "Layer Stage": "0. Unsupervised PCA",
                    "Input Dimension": "1,024 pixels",
                    "Output Dimension": "50 features",
                    "Weight Formula": "Orthogonal Projection W (1,024 × 50)",
                    "Weights": "0 (Precomputed)",
                    "Biases": "0",
                    "Parameters": "0 learnable",
                    "Activation": "Linear Projection",
                },
                {
                    "Layer Stage": "1. Hidden Layer 1",
                    "Input Dimension": "50 features",
                    "Output Dimension": "100 neurons",
                    "Weight Formula": "Linear(50, 100): 50 × 100",
                    "Weights": "5,000",
                    "Biases": "100",
                    "Parameters": "5,100",
                    "Activation": "ReLU(x) = max(0, x)",
                },
                {
                    "Layer Stage": "2. Hidden Layer 2",
                    "Input Dimension": "100 neurons",
                    "Output Dimension": "50 neurons",
                    "Weight Formula": "Linear(100, 50): 100 × 50",
                    "Weights": "5,000",
                    "Biases": "50",
                    "Parameters": "5,050",
                    "Activation": "ReLU(x) = max(0, x)",
                },
                {
                    "Layer Stage": "3. Output Layer",
                    "Input Dimension": "50 neurons",
                    "Output Dimension": "10 class logits",
                    "Weight Formula": "Linear(50, 10): 50 × 10",
                    "Weights": "500",
                    "Biases": "10",
                    "Parameters": "510",
                    "Activation": "Softmax",
                },
            ])

            st.dataframe(param_df, use_container_width=True, hide_index=True)

            render_html(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; background-color: #F0FDF4; border: 1px solid #86EFAC; border-radius: 12px; padding: 1rem 1.25rem; margin-top: 1rem;">
                    <div>
                        <div style="font-weight: 800; font-size: 0.95rem; color: #064E3B;">Total Trainable Model Footprint</div>
                        <div style="font-size: 0.8rem; color: #065F46; margin-top: 0.15rem;">
                            10,500 weight connections + 160 bias terms = <strong>10,660 learnable parameters</strong>.
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="background-color: #22C55E; color: #FFFFFF; font-weight: 800; font-size: 1.15rem; padding: 0.4rem 1rem; border-radius: 9999px;">
                            10,660 Params
                        </span>
                    </div>
                </div>
                """
            )


# -----------------------------------------------------------------------------
# Section H: Model Architecture & Methodology
# -----------------------------------------------------------------------------
st.write("")
st.markdown('<div id="about-model"></div>', unsafe_allow_html=True)
render_html(
    """
    <div class="section-heading-row" style="margin-top: 1.5rem;">
        <div>
            <div class="section-main-title">Model Architecture & Methodology</div>
            <div class="section-main-desc">
                How unsupervised Principal Component Analysis and supervised Artificial Neural Networks 
                collaborate to achieve superior generalization.
            </div>
        </div>
    </div>
    """
)

arch_col1, arch_col2 = st.columns([1, 1], gap="medium")

with arch_col1:
    with st.container(border=True):
        render_html(
            """
            <span class="arch-header-badge gray">Unsupervised Preprocessing</span>
            <div class="card-title">Principal Component Analysis (PCA)</div>
            <div class="card-desc">
                Discovers orthogonal axes of maximum variance across 1,024-dimensional pixel arrays without requiring category labels.
            </div>
            
            <div class="arch-feature-item">
                <div class="arch-feature-content">
                    <div class="arch-feature-title">Spatial Collinearity Elimination</div>
                    <div class="arch-feature-desc">Neighboring pixels in a 32×32 crop share heavy mutual redundancy. PCA decouples these redundant dimensions into independent components.</div>
                </div>
                <span class="arch-feature-badge green">Orthogonal</span>
            </div>
            
            <div class="arch-feature-item">
                <div class="arch-feature-content">
                    <div class="arch-feature-title">95.12% Dimension Compression</div>
                    <div class="arch-feature-desc">1,024 raw pixels compressed into 50 principal components while retaining dominant variance and eliminating high-frequency road noise.</div>
                </div>
                <span class="arch-feature-badge green">1024 ➔ 50</span>
            </div>
            
            <div class="arch-feature-item">
                <div class="arch-feature-content">
                    <div class="arch-feature-title">Overfitting Defense</div>
                    <div class="arch-feature-desc">Reduces first-layer parameter requirements from 102,400 weights down to only 5,000 weights, accelerating convergence on limited datasets.</div>
                </div>
                <span class="arch-feature-badge green">-95.1% Weights</span>
            </div>
            
            <div class="layer-total-footer" style="background:#F9FAFB; border-color:#E5E7EB; margin-top:0.85rem;">
                <div class="layer-total-label" style="color:#374151;">Input Compression Ratio</div>
                <div class="layer-total-value" style="color:#111827;">1,024 ➔ 50 Features</div>
            </div>
            """
        )

with arch_col2:
    with st.container(border=True):
        render_html(
            """
            <span class="arch-header-badge">Supervised Deep Learning</span>
            <div class="card-title">PyTorch Artificial Neural Network (ANN)</div>
            <div class="card-desc">
                A 3-layer PyTorch feedforward neural network engineered with non-linear ReLU activations and Cross-Entropy loss.
            </div>
            
            <div class="layer-card-stack">
                <div class="layer-card-row">
                    <div class="layer-info">
                        <div class="layer-index-badge">0</div>
                        <div>
                            <div class="layer-name-title">Input Layer (PCA Features)</div>
                            <div class="layer-tech-detail">Input Shape: (batch, 50) float32</div>
                        </div>
                    </div>
                    <span class="layer-params-pill">50 Inputs</span>
                </div>
                
                <div class="layer-card-row">
                    <div class="layer-info">
                        <div class="layer-index-badge">1</div>
                        <div>
                            <div class="layer-name-title">Hidden Layer 1 (100 Neurons)</div>
                            <div class="layer-tech-detail">Linear(50, 100) + ReLU()</div>
                        </div>
                    </div>
                    <span class="layer-params-pill">5,100 params</span>
                </div>
                
                <div class="layer-card-row">
                    <div class="layer-info">
                        <div class="layer-index-badge">2</div>
                        <div>
                            <div class="layer-name-title">Hidden Layer 2 (50 Neurons)</div>
                            <div class="layer-tech-detail">Linear(100, 50) + ReLU()</div>
                        </div>
                    </div>
                    <span class="layer-params-pill">5,050 params</span>
                </div>
                
                <div class="layer-card-row">
                    <div class="layer-info">
                        <div class="layer-index-badge">3</div>
                        <div>
                            <div class="layer-name-title">Output Layer (10 Class Logits)</div>
                            <div class="layer-tech-detail">Linear(50, 10) ➔ Softmax</div>
                        </div>
                    </div>
                    <span class="layer-params-pill">510 params</span>
                </div>
            </div>
            
            <div class="layer-total-footer" style="margin-top:0.85rem;">
                <div class="layer-total-label">Total Learnable Parameters</div>
                <div class="layer-total-value">10,660 Params</div>
            </div>
            """
        )


# -----------------------------------------------------------------------------
# Section I: Experimental Benchmark Table
# -----------------------------------------------------------------------------
st.write("")
st.markdown('<div id="benchmark"></div>', unsafe_allow_html=True)
with st.container(border=True):
    render_html(
        """
        <div class="card-title">Model Performance & Comparative Study</div>
        <div class="card-desc">
            Validation and test accuracy recorded during final project experimentation across 7,498 cropped traffic signs:
        </div>
        
        <div class="benchmark-table-wrapper">
            <table class="kiwi-benchmark-table">
                <thead>
                    <tr>
                        <th>Model Architecture</th>
                        <th>Input Features</th>
                        <th>Validation Accuracy</th>
                        <th>Visual Benchmark</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>ANN Model A</strong> (Raw Pixels)</td>
                        <td>1,024</td>
                        <td><strong>93.77%</strong></td>
                        <td>
                            <div class="bar-in-table">
                                <div class="bar-in-table-track"><div class="bar-in-table-fill" style="width: 93.77%;"></div></div>
                                <span style="font-size:0.75rem; color:#6B7280;">93.77%</span>
                            </div>
                        </td>
                        <td><span class="pill-btn-outline" style="padding:0.2rem 0.65rem; font-size:0.72rem;">Baseline</span></td>
                    </tr>
                    <tr>
                        <td><strong>ANN Model B</strong> (Raw Pixels)</td>
                        <td>1,024</td>
                        <td><strong>92.25%</strong></td>
                        <td>
                            <div class="bar-in-table">
                                <div class="bar-in-table-track"><div class="bar-in-table-fill" style="width: 92.25%;"></div></div>
                                <span style="font-size:0.75rem; color:#6B7280;">92.25%</span>
                            </div>
                        </td>
                        <td><span class="pill-btn-outline" style="padding:0.2rem 0.65rem; font-size:0.72rem;">Baseline</span></td>
                    </tr>
                    <tr class="champion-row">
                        <td><strong>PCA + ANN (Our System)</strong></td>
                        <td><strong>50 (PCA)</strong></td>
                        <td><strong style="color:#065F46;">96.44%</strong></td>
                        <td>
                            <div class="bar-in-table">
                                <div class="bar-in-table-track"><div class="bar-in-table-fill green" style="width: 96.44%;"></div></div>
                                <span style="font-size:0.75rem; font-weight:700; color:#065F46;">96.44%</span>
                            </div>
                        </td>
                        <td><span class="pill-btn-green" style="padding:0.2rem 0.75rem; font-size:0.72rem;">🏆 Deployed Champion</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="champion-hero-card">
            <div class="champion-hero-left">
                <div class="champion-trophy-icon">🏆</div>
                <div>
                    <div class="champion-hero-title">Final Held-Out Test Evaluation: 96.63%</div>
                    <div class="champion-hero-desc">
                        1,089 correct predictions out of 1,127 unseen test crops, proving that PCA dimensionality reduction + ANN significantly outperforms direct raw-pixel modeling on real Cambodian road signs.
                    </div>
                </div>
            </div>
            <div class="champion-hero-score">
                <div class="champion-score-number">96.63%</div>
                <div class="champion-score-label">Test Split Accuracy</div>
            </div>
        </div>
        
        <div style="font-size:0.74rem; color:#9CA3AF; margin-top:0.85rem; text-align:center;">
            *Note: Benchmark figures reflect previously recorded offline academic evaluation (70% train, 15% validation, 15% test) and remain distinct from live interactive predictions.
        </div>
        """
    )


# -----------------------------------------------------------------------------
# Section J: Supported Classes Grid
# -----------------------------------------------------------------------------
st.write("")
st.markdown('<div id="classes"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-heading-row" style="margin-top: 1.5rem;">
        <div>
            <div class="section-main-title">Supported Cambodian Traffic Signs</div>
            <div class="section-main-desc">
                The 10 specific Cambodian road signs recognized by this trained neural network.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

grid_c = st.columns(5, gap="small")
for iid in range(10):
    c_info = catalog.get(iid, {})
    col_pos = iid % 5
    with grid_c[col_pos]:
        with st.container(border=True):
            icon_path = ROOT_DIR / c_info.get("sample_image", "")
            if icon_path.is_file():
                st.image(str(icon_path), use_container_width=True)
            else:
                st.markdown(
                    f"<div style='height:75px; display:flex; align-items:center; justify-content:center; background:#F9FAFB; border-radius:12px;'>Sign #{c_info.get('original_id')}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div class="class-cell-name">{c_info.get('name')}</div>
                    <div class="class-cell-id">Class {c_info.get('original_id')} (Idx {iid})</div>
                    <div style="font-size:0.72rem; color:#059669; font-weight:600; margin-top:0.15rem;">{c_info.get('category')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# Section K: Educational Disclaimer & Kiwi Minimalist Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="kiwi-footer">
        <p>
            <strong>TrafficSign AI — Cambodian Traffic Sign Recognition System</strong><br>
            University Machine Learning Final Project • Built with Python, PyTorch, Scikit-learn, and Streamlit
        </p>
        <p style="font-size: 0.78rem; color: #9CA3AF; max-width: 620px; margin: 0.6rem auto 0 auto;">
            <strong>Educational Disclaimer:</strong> This application was developed strictly for university academic research and machine learning evaluation. It classifies single, closely cropped signs and must not be used for autonomous vehicle control or safety-critical driving systems.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
