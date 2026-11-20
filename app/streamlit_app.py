"""Streamlit prototype for single-image industrial anomaly inspection.

Run with:
    streamlit run app/streamlit_app.py

The app offers exactly the six categories in the frozen project scope and only
the models trained for the selected category. Categories outside the scope are
never loaded or offered. It never falls back to another category's model or
thresholds.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from industrial_defect.categories import (  # noqa: E402
    out_of_scope_categories,
    supported_categories,
)
from industrial_defect.config import load_config  # noqa: E402
from industrial_defect.inference import (  # noqa: E402
    DISCLAIMER,
    IndustrialInspector,
    InferenceError,
)
from industrial_defect.inventory import category_inventory, category_status  # noqa: E402
from industrial_defect.training import available_models  # noqa: E402

st.set_page_config(
    page_title="Industrial Defect Inspection",
    page_icon=None,
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_config() -> dict:
    return load_config()


@st.cache_resource(show_spinner="Loading model and calibrated thresholds...")
def get_inspector(model_name: str, category: str) -> IndustrialInspector:
    return IndustrialInspector(get_config(), model_name, category)


def png_bytes(array) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def build_zip(result, metadata: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("heatmap.png", png_bytes(result.heatmap))
        archive.writestr("overlay.png", png_bytes(result.overlay))
        archive.writestr(
            "predicted_mask.png",
            png_bytes(result.predicted_mask.astype("uint8") * 255),
        )
        archive.writestr("result.json", metadata)
        archive.writestr(
            "README.txt",
            "Industrial defect inspection prototype\n"
            f"{DISCLAIMER}\n"
            "Heatmap/overlay are normalised per image; the predicted mask uses the "
            "calibrated pixel threshold.\n",
        )
    return buffer.getvalue()


config = get_config()
supported = supported_categories(config)
out_of_scope = out_of_scope_categories(config)
inventory = category_inventory(config, supported)
trained = [name for name, status in inventory.items() if status["trained_models"]]

st.title("Industrial Defect Inspection")
st.caption(
    "Unsupervised anomaly detection and localisation (PatchCore / PaDiM) trained only on "
    "defect-free images. Research prototype, not a certified production system."
)

with st.sidebar:
    st.header("Configuration")
    if not trained:
        st.error(
            "No supported category has both a trained artifact and calibrated thresholds. "
            "Run scripts/04_train.py and scripts/05_calibrate.py first."
        )
        st.stop()
    default_category = str(config["data"]["category"])
    category = st.selectbox(
        "MVTec category (frozen scope)",
        supported,
        index=supported.index(default_category) if default_category in supported else 0,
        help="Only the six supported categories are offered; the other nine are out of scope.",
    )
    models = available_models(config, category)
    if not models:
        st.error(
            f"No trained artifact and calibrated thresholds for category '{category}'. "
            "This app never substitutes another category's model."
        )
        st.stop()
    model_name = st.selectbox("Model", models)
    status = category_status(config, category)
    st.caption(
        f"Category `{category}` — "
        f"{'evaluated on the test set' if status['evaluated_models'] else 'not yet evaluated'} "
        f"({', '.join(status['evaluated_models']) or 'no metrics'})."
    )
    st.divider()
    st.caption(
        "Thresholds are calibrated on held-out normal training images of the selected "
        "category only. Test anomalies were never used for fitting or calibration."
    )

with st.expander("Category availability (six supported categories)", expanded=False):
    table = pd.DataFrame(
        [
            {
                "category": name,
                "dataset": "available" if status["data_available"] else "missing",
                "trained models": ", ".join(status["trained_models"]) or "-",
                "evaluated models": ", ".join(status["evaluated_models"]) or "-",
            }
            for name, status in inventory.items()
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(
        "The project scope is frozen at these six categories. "
        f"The other nine MVTec AD categories are out of scope and are not loaded: "
        f"{', '.join(out_of_scope)}. "
        f"{len(trained)} of {len(supported)} supported categories have trained models."
    )

uploaded = st.file_uploader(
    "Upload an inspection image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded is None:
    st.info(
        "Upload a single image to obtain an image-level PASS/DEFECT decision, an anomaly "
        "heatmap, an overlay and a thresholded defect mask. Results can be downloaded as "
        "PNG/JSON or a ZIP bundle."
    )
    st.stop()

try:
    image = Image.open(uploaded)
    image.load()
except (OSError, ValueError) as exc:
    st.error(f"Cannot read the uploaded image: {exc}")
    st.stop()

try:
    inspector = get_inspector(model_name, category)
    result = inspector.inspect(image, source_name=uploaded.name)
except InferenceError as exc:
    st.error(str(exc))
    st.stop()
except FileNotFoundError as exc:
    st.error(f"Missing artifact: {exc}")
    st.stop()

descriptor = inspector.describe()
left, middle, right, far = st.columns(4)
left.metric("Anomaly score", f"{result.score:.3f}")
middle.metric("Image threshold", f"{result.image_threshold:.3f}")
right.metric("Pixel threshold", f"{result.pixel_threshold:.3f}")
far.metric("Predicted defect area", f"{result.defect_area_fraction * 100:.2f}%")

if result.decision == "defect":
    st.error(
        f"DEFECT - image score {result.score:.3f} is above the calibrated threshold "
        f"{result.image_threshold:.3f}."
    )
else:
    st.success(
        f"PASS - image score {result.score:.3f} is below the calibrated threshold "
        f"{result.image_threshold:.3f}."
    )
st.caption(
    f"Decision produced by the `{category}` / `{model_name}` model. The image-level "
    "decision uses the image score; the predicted mask uses the pixel threshold and can "
    "be non-empty even when the image decision is PASS."
)

columns = st.columns(3)
columns[0].image(image.convert("RGB"), caption="Input image", use_column_width=True)
columns[1].image(result.heatmap, caption="Anomaly heatmap (per-image normalised)", use_column_width=True)
columns[2].image(result.overlay, caption="Overlay", use_column_width=True)

metadata = json.dumps(
    {**result.to_dict(), "provenance": descriptor}, indent=2, ensure_ascii=False
).encode("utf-8")
download_columns = st.columns(4)
download_columns[0].download_button(
    "Overlay PNG", png_bytes(result.overlay), file_name="overlay.png", mime="image/png"
)
download_columns[1].download_button(
    "Heatmap PNG", png_bytes(result.heatmap), file_name="heatmap.png", mime="image/png"
)
download_columns[2].download_button(
    "Result JSON", metadata, file_name="result.json", mime="application/json"
)
download_columns[3].download_button(
    "ZIP bundle", build_zip(result, metadata), file_name="inspection.zip", mime="application/zip"
)

with st.expander("Model and threshold provenance"):
    st.json(descriptor)

st.divider()
st.caption(DISCLAIMER)
st.caption(
    f"Processed at {result.processed_size[0]}x{result.processed_size[1]} from an original "
    f"{result.original_size[0]}x{result.original_size[1]} image in {result.latency_ms:.0f} ms "
    f"on {descriptor['device']}."
)
