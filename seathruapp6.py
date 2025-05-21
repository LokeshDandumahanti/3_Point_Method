import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="3-Point Underwater Color Corrector", layout="centered")
st.title("🌊 Underwater Image Color Correction")

# --- Color Balancing Function ---
def gray_world_white_balance(img):
    img = img.astype(np.float32)
    avg_b, avg_g, avg_r = [np.mean(img[:, :, c]) for c in range(3)]
    avg_gray = (avg_b + avg_g + avg_r) / 3
    scale = [avg_gray / avg_b, avg_gray / avg_g, avg_gray / avg_r]

    for c in range(3):
        img[:, :, c] *= scale[c]

    return np.clip(img, 0, 255).astype(np.uint8)

# --- 3-Point Correction Function ---
def apply_3_point_color_correction(image, shadow_shift, midtone_shift, highlight_shift):
    image = image.astype(np.float32)
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    shadow_mask = gray < 85
    midtone_mask = (gray >= 85) & (gray < 170)
    highlight_mask = gray >= 170

    for c in range(3):
        image[:, :, c][shadow_mask] += shadow_shift[c]
        image[:, :, c][midtone_mask] += midtone_shift[c]
        image[:, :, c][highlight_mask] += highlight_shift[c]

    return np.clip(image, 0, 255).astype(np.uint8)

# --- Upload Section ---
uploaded_file = st.file_uploader("📤 Upload an underwater image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="📷 Original Image", use_column_width=True)

    # --- Auto Color Balance ---
    apply_balance = st.checkbox("🧪 Auto Color Balance (Neutralize Blue/Green Tint)", value=True)

    if apply_balance:
        image_bgr = gray_world_white_balance(image_bgr)

    # --- 3-Point Sliders ---
    st.markdown("### 🎛️ 3-Point Color Shift Controls")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Shadows**")
        sr = st.slider("Red", -50, 50, 0, key="sr")
        sg = st.slider("Green", -50, 50, 0, key="sg")
        sb = st.slider("Blue", -50, 50, 0, key="sb")

    with col2:
        st.markdown("**Midtones**")
        mr = st.slider("Red", -50, 50, 0, key="mr")
        mg = st.slider("Green", -50, 50, 0, key="mg")
        mb = st.slider("Blue", -50, 50, 0, key="mb")

    with col3:
        st.markdown("**Highlights**")
        hr = st.slider("Red", -50, 50, 0, key="hr")
        hg = st.slider("Green", -50, 50, 0, key="hg")
        hb = st.slider("Blue", -50, 50, 0, key="hb")

    shadow_shift = [sb, sg, sr]
    midtone_shift = [mb, mg, mr]
    highlight_shift = [hb, hg, hr]

    corrected = apply_3_point_color_correction(image_bgr.copy(), shadow_shift, midtone_shift, highlight_shift)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

    st.image(corrected_rgb, caption="✅ Corrected Image", use_column_width=True)

    # --- Download Button ---
    corrected_pil = Image.fromarray(corrected_rgb)
    buf = BytesIO()
    corrected_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Corrected Image",
        data=byte_im,
        file_name="corrected_image.png",
        mime="image/png"
    )
