"""
Background Remover — Streamlit Web App

A single-file Streamlit application that:
  • Uploads images from your device
  • Removes the background with adjustable options (model, alpha matting,
    mask post-processing, edge-softness slider)
  • Shows a side-by-side preview (original vs. result) with a transparency
    checkerboard
  • Replaces the background with a solid colour or another image
  • Downloads the result as PNG / JPEG / WebP

Install dependencies:
    pip install -r requirements.txt

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

# --------------------------------------------------------------------------- #
# Optional heavy dependency — keep the app launchable even if it's missing
# --------------------------------------------------------------------------- #
try:
    from rembg import new_session, remove
    REMBG_AVAILABLE = True
    REMBG_ERROR = None
except Exception as _e:                       # pragma: no cover
    REMBG_AVAILABLE = False
    REMBG_ERROR = str(_e)


# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Background Remover",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODELS = [
    "isnet-general-use",    # fast + very good quality  (recommended default)
    "birefnet-general",     # best all-round quality (slower)
    "birefnet-portrait",    # people / headshots
    "birefnet-hrsod",       # high-resolution salient objects
    "u2net",                # classic
    "u2net_human_seg",      # people
    "u2netp",               # fastest, lower quality
]

BG_OPTIONS = [
    "Transparent",
    "White",
    "Black",
    "Gray",
    "Custom color",
    "Custom image",
]

UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]


# --------------------------------------------------------------------------- #
# Cached resources / data
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_session(model_name: str):
    """Load (and cache) a rembg inference session per model name."""
    return new_session(model_name)


@st.cache_data(show_spinner=False, max_entries=6)
def run_removal(
    image_bytes: bytes,
    model_name: str,
    matting: bool,
    postprocess: bool,
    erode_size: int,
) -> bytes:
    """
    Run the segmentation model and return the RGBA result as PNG bytes.

    Cached on (image, model, matting, postprocess, erode) so that changing
    the background or the edge-softness slider never re-runs the model.
    """
    src = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    session = get_session(model_name)

    result = remove(
        src,
        session=session,
        alpha_matting=matting,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=erode_size,
        post_process_mask=postprocess,
    )
    if not isinstance(result, Image.Image):
        result = Image.open(result)
    result = result.convert("RGBA")

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def apply_feather(rgba: Image.Image, radius: float) -> Image.Image:
    """Soften the alpha channel with a Gaussian blur."""
    if radius <= 0.05:
        return rgba
    r, g, b, a = rgba.split()
    a = a.filter(ImageFilter.GaussianBlur(radius))
    return Image.merge("RGBA", (r, g, b, a))


def checkerboard(size: tuple[int, int], square: int = 12) -> Image.Image:
    """Fast tiled transparency checkerboard."""
    w, h = size
    tile = Image.new("RGB", (square * 2, square * 2), (235, 235, 235))
    d = ImageDraw.Draw(tile)
    d.rectangle([0, 0, square - 1, square - 1], fill=(205, 205, 205))
    d.rectangle(
        [square, square, square * 2 - 1, square * 2 - 1], fill=(205, 205, 205)
    )
    bg = Image.new("RGB", (w, h))
    step = square * 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            bg.paste(tile, (x, y))
    return bg


def fit_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale for on-screen preview only (never for the download)."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    s = max_side / max(w, h)
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def preview_on_checker(rgba: Image.Image, max_side: int = 1400) -> Image.Image:
    """Composite an RGBA image over a checkerboard for display."""
    disp = fit_max_side(rgba, max_side)
    bg = checkerboard(disp.size)
    bg.paste(disp, (0, 0), disp)
    return bg


def solid_bg(rgba: Image.Image, color: tuple[int, int, int, int]) -> Image.Image:
    bg = Image.new("RGBA", rgba.size, color)
    return Image.alpha_composite(bg, rgba)


def image_bg(rgba: Image.Image, bg_img: Image.Image) -> Image.Image:
    """Cover-fit the background image behind the cut-out subject."""
    bg = bg_img.convert("RGBA")
    scale = max(rgba.width / bg.width, rgba.height / bg.height)
    new_size = (
        max(1, round(bg.width * scale)),
        max(1, round(bg.height * scale)),
    )
    bg = bg.resize(new_size, Image.LANCZOS)
    left = (bg.width - rgba.width) // 2
    top = (bg.height - rgba.height) // 2
    bg = bg.crop((left, top, left + rgba.width, top + rgba.height))
    return Image.alpha_composite(bg, rgba)


def hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i: i + 2], 16) for i in (0, 2, 4))
    return (r, g, b, 255)


def encode_for_download(img: Image.Image, fmt: str) -> tuple[bytes, str, str]:
    """Return (bytes, mime, extension) for the chosen format."""
    buf = io.BytesIO()
    fmt = fmt.upper()

    if fmt == "JPEG":
        if img.mode == "RGBA":
            flat = Image.new("RGB", img.size, (255, 255, 255))
            flat.paste(img, (0, 0), img)
            img = flat
        img.save(buf, "JPEG", quality=95, subsampling=0, optimize=True)
        return buf.getvalue(), "image/jpeg", "jpg"

    if fmt == "WEBP":
        img.save(buf, "WEBP", quality=95, method=6)
        return buf.getvalue(), "image/webp", "webp"

    img.save(buf, "PNG", optimize=True)
    return buf.getvalue(), "image/png", "png"


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("✂️ Background Remover")
st.caption(
    "Remove the background from any image, fine-tune the edges, "
    "swap in a new background and download the result — all in your browser."
)

if not REMBG_AVAILABLE:
    st.error(
        "**Missing dependency:** could not import `rembg`.\n\n"
        f"```\n{REMBG_ERROR}\n```\n"
        "Install it with `pip install rembg pillow onnxruntime` "
        "and restart the app."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Sidebar — controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Processing options")

    model_name = st.selectbox(
        "Segmentation model",
        MODELS,
        index=0,
        help=(
            "`isnet-general-use` is a great speed/quality balance. "
            "`birefnet-*` gives the best quality but is slower. "
            "`u2netp` is the fastest, lowest quality."
        ),
    )

    postprocess = st.checkbox(
        "Post-process mask (cleanup)",
        value=True,
        help="Removes small artefacts and jagged edges from the mask.",
    )

    with st.expander("Alpha matting (hair / fur / fine detail)"):
        matting = st.checkbox(
            "Enable alpha matting",
            value=False,
            help=(
                "Produces much finer edges for hair and fur. "
                "Slower and needs `pymatting` installed."
            ),
        )
        erode_size = st.slider(
            "Matting erode size",
            min_value=0,
            max_value=30,
            value=10,
            step=1,
            disabled=not matting,
            help="Changing this re-runs the model.",
        )
        st.caption("⚠️ Changing any option above re-runs the segmentation model.")

    st.divider()

    st.header("🎚️ Edge softness")
    feather = st.slider(
        "Edge softness / intensity",
        min_value=0.0,
        max_value=5.0,
        value=0.6,
        step=0.1,
        help="Blurs the alpha channel. Applied instantly — no re-processing.",
    )

    st.divider()

    st.header("🖼️ Background")
    bg_mode = st.selectbox("Replace background with", BG_OPTIONS, index=0)

    custom_color = (255, 255, 255, 255)
    custom_bg_image: Image.Image | None = None

    if bg_mode == "Custom color":
        picked = st.color_picker("Pick a colour", "#ffffff")
        custom_color = hex_to_rgba(picked)
        st.caption(f"Solid #{picked.lstrip('#').lower()}")

    elif bg_mode == "Custom image":
        bg_file = st.file_uploader(
            "Background image",
            type=UPLOAD_TYPES,
            key="bg_upload",
        )
        if bg_file is not None:
            try:
                custom_bg_image = Image.open(
                    io.BytesIO(bg_file.getvalue())
                ).convert("RGBA")
                st.caption(f"Background: {bg_file.name}")
            except Exception as e:
                st.error(f"Could not load background image: {e}")

    st.divider()

    st.header("💾 Download")
    out_format = st.radio(
        "Format",
        ["PNG", "JPEG", "WEBP"],
        index=0,
        horizontal=True,
        help="PNG and WEBP keep transparency. JPEG is flattened onto white.",
    )


# --------------------------------------------------------------------------- #
# Main — upload
# --------------------------------------------------------------------------- #
uploaded = st.file_uploader(
    "Upload an image",
    type=UPLOAD_TYPES,
    help="PNG, JPG, JPEG, WEBP, BMP or TIFF — up to 200 MB.",
)

if uploaded is None:
    st.info("👆 Upload an image to get started.")
    st.stop()

raw_bytes = uploaded.getvalue()

try:
    input_image = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
except Exception as e:
    st.error(f"Could not open that image: {e}")
    st.stop()

w, h = input_image.size
st.caption(f"**{uploaded.name}** — {w} × {h} px")


# --------------------------------------------------------------------------- #
# Run the model (cached)
# --------------------------------------------------------------------------- #
try:
    with st.spinner("Running segmentation model… (first run may download weights)"):
        result_png = run_removal(
            raw_bytes,
            model_name,
            bool(matting),
            bool(postprocess),
            int(erode_size) if matting else 10,
        )
except Exception as e:
    st.error(f"Processing failed: {e}")
    st.stop()

raw_output = Image.open(io.BytesIO(result_png)).convert("RGBA")
output_image = apply_feather(raw_output, float(feather))


# --------------------------------------------------------------------------- #
# Composite the chosen background
# --------------------------------------------------------------------------- #
if bg_mode == "White":
    final = solid_bg(output_image, (255, 255, 255, 255))
elif bg_mode == "Black":
    final = solid_bg(output_image, (0, 0, 0, 255))
elif bg_mode == "Gray":
    final = solid_bg(output_image, (128, 128, 128, 255))
elif bg_mode == "Custom color":
    final = solid_bg(output_image, custom_color)
elif bg_mode == "Custom image" and custom_bg_image is not None:
    final = image_bg(output_image, custom_bg_image)
else:
    final = output_image          # Transparent (or image not yet chosen)


# --------------------------------------------------------------------------- #
# Preview — side by side
# --------------------------------------------------------------------------- #
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Original")
    st.image(input_image, use_container_width=True)

with col_right:
    st.subheader("Result")
    if final.mode == "RGBA":
        st.image(preview_on_checker(final), use_container_width=True)
    else:
        st.image(final, use_container_width=True)


# --------------------------------------------------------------------------- #
# Mask preview + stats
# --------------------------------------------------------------------------- #
with st.expander("🔍 Mask / alpha preview"):
    alpha = raw_output.getchannel("A")
    m1, m2 = st.columns(2)
    with m1:
        st.caption("Alpha mask")
        st.image(alpha, use_container_width=True, clamp=True)
    with m2:
        st.caption("Feathered alpha")
        st.image(output_image.getchannel("A"), use_container_width=True, clamp=True)

coverage = sum(raw_output.getchannel("A").getdata()) / (255 * w * h) * 100
st.caption(f"Subject coverage: **{coverage:.1f}%** of the frame · Model: `{model_name}`")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
st.divider()

stem = Path(uploaded.name).stem
payload, mime, ext = encode_for_download(final, out_format)

st.download_button(
    label=f"⬇️  Download {out_format}",
    data=payload,
    file_name=f"{stem}_nobg.{ext}",
    mime=mime,
    use_container_width=True,
    type="primary",
)

st.caption(
    "All processing happens on the server running this app. "
    "Nothing is stored — images live in memory only for your session."
)