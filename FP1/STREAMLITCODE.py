import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import streamlit as st
from skimage import exposure, filters, io, morphology

# ======================================================================
# KONFIGURASI
# ======================================================================
# Taruh semua file citra di folder "images" (sejajar dengan app.py)
IMAGE_DIR = "images"
IMAGE_FILES = {
    "ISIC_0010447": "ISIC_0010447.jpg",
    "ISIC_0011157": "ISIC_0011157.jpg",
    "ISIC_0010571": "ISIC_0010571.jpg",
    "BCC_260": "BCC_260.jpg",
    "BCC_32": "BCC_32.jpg",
    "BCC_91": "BCC_91.jpg",
}

ENHANCEMENTS = [
    "Histogram Equalization (HE)",
    "Adaptive Histogram Equalization (AHE)",
    "Contrast Stretching",
    "Image Negative",
]
CLIP_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05]
SIZES = [3, 5, 7]
GAUSS_SIGMA = {3: 0.5, 5: 1.0, 7: 1.5}

st.set_page_config(page_title="Pengolahan Citra Medika", layout="wide")

# ======================================================================
# STYLE
# ======================================================================
st.markdown(
    """
    <style>
    .stApp { background-color: #ffd6d6; }
    .header {
        background: #f08a9b; color: white; text-align: center;
        font-size: 28px; font-weight: 800; letter-spacing: 1px;
        padding: 10px 20px; border-radius: 30px; margin-bottom: 18px;
    }
    .pill {
        display: inline-block; background: #f08a9b; color: white;
        font-weight: 800; padding: 3px 16px; border-radius: 20px;
        margin: 18px 0 8px 0; font-size: 15px;
    }
    div[data-baseweb="select"] > div { background-color: #9b1c31; color: white; }
    div[data-baseweb="select"] svg { fill: white; }
    div[data-baseweb="select"] span { color: white; }
    label, .stMarkdown p { color: #3a0a12; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
# FUNGSI PENGOLAHAN CITRA
# ======================================================================
@st.cache_data
def load_image(name):
    path = os.path.join(IMAGE_DIR, IMAGE_FILES[name])
    image = io.imread(path)
    if image.ndim == 2:                       # grayscale -> RGB
        image = np.stack([image] * 3, axis=2)
    return image[:, :, :3]                    # buang alpha jika ada


def histogram_equalization(img):
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    return np.clip(cdf[img] * 255, 0, 255).astype(np.uint8)


def adaptive_he(img, clip):
    result = exposure.equalize_adapthist(img, clip_limit=clip)
    return np.clip(np.round(result * 255), 0, 255).astype(np.uint8)


def contrast_stretching(img):
    c, d = int(img.min()), int(img.max())
    if d == c:
        return img.copy()
    out = (img.astype(float) - c) * (255 / (d - c))
    return np.clip(out, 0, 255).astype(np.uint8)


def image_negative(img):
    return 255 - img


def apply_enhancement(blue, method, clip):
    if method == ENHANCEMENTS[0]:
        return histogram_equalization(blue)
    if method == ENHANCEMENTS[1]:
        return adaptive_he(blue, clip)
    if method == ENHANCEMENTS[2]:
        return contrast_stretching(blue)
    return image_negative(blue)


def gaussian_kernel(size, sigma):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    w = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return w / w.sum()


def laplacian_kernel(size):
    sigma = size / 6
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx**2 + yy**2
    w = (1 - r2 / (2 * sigma**2)) * np.exp(-r2 / (2 * sigma**2))
    w = w - w.mean()
    return w * 4 / w[w > 0].sum()


def calculate_mse(ref, res):
    return np.mean((ref.astype(float) - res.astype(float)) ** 2)


def calculate_psnr(mse):
    return float("inf") if mse == 0 else 10 * np.log10((255**2) / mse)


def to_u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


@st.cache_data
def run_filters_and_mask(name):
    """Semua proses di bagian FILTER dan MASKING (berdasarkan Contrast Stretching blue channel)."""
    image = load_image(name)
    cs = contrast_stretching(image[:, :, 2])
    f = cs.astype(float)

    results = {"Mean": [], "Median": [], "Gaussian": [], "Laplacian": []}
    for s in SIZES:
        results["Mean"].append(to_u8(ndi.convolve(f, np.full((s, s), 1 / s**2))))
        results["Median"].append(ndi.median_filter(cs, size=s))
        results["Gaussian"].append(to_u8(ndi.convolve(f, gaussian_kernel(s, GAUSS_SIGMA[s]))))
        results["Laplacian"].append(to_u8(f + ndi.convolve(f, laplacian_kernel(s))))

    rows = []
    for fname, res in results.items():
        for s, r in zip(SIZES, res):
            mse = calculate_mse(cs, r)
            rows.append({
                "Filter": fname,
                "Kernel": f"{s}x{s}",
                "MSE": round(mse, 3),
                "PSNR (dB)": round(calculate_psnr(mse), 3),
            })
    df_metrics = pd.DataFrame(rows)

    # Masking: Gaussian 3x3 -> Otsu -> binary -> bersihkan
    gauss = results["Gaussian"][0]
    threshold = filters.threshold_otsu(gauss)
    binary = gauss < threshold
    min_size = 50 if image.shape[0] == 28 else 300
    large = morphology.remove_small_objects(binary, min_size=min_size)
    mask = np.logical_not(
        morphology.remove_small_objects(np.logical_not(large), min_size=min_size)
    )
    masked = np.where(mask[:, :, np.newaxis], image, 0)

    return {
        "cs": cs, "results": results, "df": df_metrics,
        "gauss": gauss, "threshold": threshold, "binary": binary,
        "mask": mask, "masked": masked,
    }


# ======================================================================
# HELPER TAMPILAN
# ======================================================================
def section(title):
    st.markdown(f'<div class="pill">{title}</div>', unsafe_allow_html=True)


def show_images(imgs, captions):
    cols = st.columns(len(imgs))
    for col, img, cap in zip(cols, imgs, captions):
        col.image(img, caption=cap, width="stretch", clamp=True)


def filter_row(label, cs, res):
    show_images(
        [cs] + res,
        ["Contrast Stretching"] + [f"{label} {s}x{s}" for s in SIZES],
    )


# ======================================================================
# HEADER + DROPDOWN
# ======================================================================
st.markdown(
    '<div class="header">PENGOLAHAN CITRA MEDIKA IMAGE ENHANCEMENT</div>',
    unsafe_allow_html=True,
)

available = [n for n, fn in IMAGE_FILES.items()
             if os.path.exists(os.path.join(IMAGE_DIR, fn))]
if not available:
    st.error(f"Tidak ada citra di folder '{IMAGE_DIR}/'. Taruh file citra di sana, "
             "lalu sesuaikan IMAGE_FILES di app.py.")
    st.stop()

dd1, dd2 = st.columns(2)
name = dd1.selectbox("Pilih citra", available)
method = dd2.selectbox("Pilih enhancement", ENHANCEMENTS)

image = load_image(name)
blue = image[:, :, 2]

# ======================================================================
# CITRA ASLI
# ======================================================================
section("CITRA ASLI")
c_left, c_right = st.columns([1, 3])
c_left.image(image, caption=name, width="stretch")

fig, axes = plt.subplots(1, 4, figsize=(14, 2.8))
for ch, color in enumerate(["red", "green", "blue"]):
    axes[0].hist(image[:, :, ch].ravel(), bins=256, range=(0, 256),
                 color=color, alpha=0.5, label=color.capitalize())
axes[0].set_title("Histogram RGB")
axes[0].legend(fontsize=7)
for ax, ch, color in zip(axes[1:], range(3), ["red", "green", "blue"]):
    ax.hist(image[:, :, ch].ravel(), bins=256, range=(0, 256), color=color)
    ax.set_title(f"Channel {color.capitalize()}")
for ax in axes:
    ax.set_xlabel("Intensitas Piksel", fontsize=8)
    ax.set_ylabel("Jumlah Piksel", fontsize=8)
    ax.tick_params(labelsize=7)
plt.tight_layout()
c_right.pyplot(fig)
plt.close(fig)

show_images(
    [image, image[:, :, 0], image[:, :, 1], image[:, :, 2]],
    ["Citra Asli", "Red Channel", "Green Channel", "Blue Channel"],
)

# ======================================================================
# ENHANCEMENT
# ======================================================================
section("ENHANCEMENT")
clip = CLIP_VALUES[0]
if method == ENHANCEMENTS[1]:
    clip = st.select_slider("clip_limit (AHE)", options=CLIP_VALUES, value=CLIP_VALUES[0])

enhanced = apply_enhancement(blue, method, clip)

e_left, e_right = st.columns([1, 3])
e_left.image(enhanced, caption=method, width="stretch", clamp=True)

h_blue, _ = np.histogram(blue.ravel(), bins=256, range=(0, 256))
h_enh, _ = np.histogram(enhanced.ravel(), bins=256, range=(0, 256))
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.plot(h_blue, label="Blue Channel", color="blue")
ax.plot(h_enh, label=method, color="crimson")
ax.set_title(f"{name} - Histogram Blue Channel vs {method}")
ax.set_xlabel("Intensitas Piksel")
ax.set_ylabel("Jumlah Piksel")
ax.legend()
plt.tight_layout()
e_right.pyplot(fig)
plt.close(fig)

# ======================================================================
# FILTER
# ======================================================================
data = run_filters_and_mask(name)
cs = data["cs"]

section("FILTER")
filter_row("Mean Filter", cs, data["results"]["Mean"])
filter_row("Median", cs, data["results"]["Median"])
filter_row("Gaussian", cs, data["results"]["Gaussian"])
filter_row("Laplacian", cs, data["results"]["Laplacian"])

st.markdown("**Perhitungan MSE dan PSNR** (terhadap citra Contrast Stretching)")
st.dataframe(data["df"], width="stretch", hide_index=True)

# ======================================================================
# MASKING
# ======================================================================
section("MASKING")
m_left, m_right = st.columns([2, 3])

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].imshow(data["gauss"], cmap="gray", vmin=0, vmax=255)
axes[0].contour(data["gauss"], [data["threshold"]])
axes[0].set_title("Gaussian 3x3 + Contour Threshold", fontsize=9)
axes[1].imshow(data["binary"], cmap="gray")
axes[1].set_title("Binary Image", fontsize=9)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
m_left.pyplot(fig)
plt.close(fig)

with m_right:
    show_images(
        [image, data["mask"].astype(np.uint8) * 255, data["masked"]],
        ["Citra Asli", "Mask Akhir", "Hasil Masking"],
    )
