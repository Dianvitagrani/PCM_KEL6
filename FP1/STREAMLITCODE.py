import base64
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
# Folder "images" dicari relatif terhadap lokasi app.py (bukan folder kerja),
# supaya tetap jalan di Streamlit Cloud walau app.py ada di dalam subfolder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
IMAGE_NAMES = ["ISIC_0010447", "ISIC_0011157", "ISIC_0010571",
               "BCC_260", "BCC_32", "BCC_91"]


def find_images():
    """Cocokkan nama citra dengan file di folder images (abaikan huruf besar/kecil & ekstensi)."""
    if not os.path.isdir(IMAGE_DIR):
        return {}
    files = {os.path.splitext(f)[0].lower(): f for f in os.listdir(IMAGE_DIR)}
    return {n: files[n.lower()] for n in IMAGE_NAMES if n.lower() in files}


IMAGE_FILES = find_images()

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


def logo_html():
    """Cari file logo*.png/jpg/webp di folder app.py atau images/, tampilkan di kiri header."""
    mimes = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".webp": "image/webp", ".svg": "image/svg+xml"}
    for folder in (BASE_DIR, IMAGE_DIR):
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(f)
            if stem.lower().startswith("logo") and ext.lower() in mimes:
                with open(os.path.join(folder, f), "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                return (f'<img src="data:{mimes[ext.lower()]};base64,{b64}" '
                        'style="height:48px;position:absolute;left:20px;top:50%;'
                        'transform:translateY(-50%);">')
    return ""


# ======================================================================
# STYLE
# ======================================================================
st.markdown(
    """
    <style>
    .stApp { background-color: #ffd6d6; }
    .header {
        position: relative; background: #f08a9b; color: white; text-align: center;
        font-size: 28px; font-weight: 800; letter-spacing: 1px;
        padding: 10px 20px; border-radius: 30px; margin-bottom: 18px;
    }
    .pill {
        display: inline-block; background: #f08a9b; color: white;
        font-weight: 800; padding: 3px 16px; border-radius: 20px;
        margin: 18px 0 8px 0; font-size: 15px;
    }
    /* Dropdown (Streamlit versi baru) */
    [data-testid="stSelectbox"] div[role="group"] {
        background-color: #9E182B !important; border-color: #9E182B !important;
    }
    [data-testid="stSelectbox"] input {
        color: white !important; -webkit-text-fill-color: white !important;
    }
    [data-testid="stSelectbox"] svg { fill: white !important; color: white !important; }
    /* Daftar pilihan saat dropdown dibuka */
    [data-testid="stSelectboxVirtualDropdown"],
    [data-testid="stSelectboxVirtualDropdown"] [role="listbox"] {
        background-color: #9E182B !important;
    }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"],
    [data-testid="stSelectboxVirtualDropdown"] [role="option"] * {
        color: white !important; background-color: transparent !important;
    }
    [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
    [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] {
        background-color: #c23a4d !important;
    }
    /* Dropdown (Streamlit versi lama) */
    div[data-baseweb="select"] > div {
        background-color: #9E182B !important; border-color: #9E182B !important;
    }
    div[data-baseweb="select"] * { color: white !important; }
    div[data-baseweb="select"] svg { fill: white !important; }
    div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {
        background-color: #9E182B !important; color: white !important;
    }
    /* Teks hitam: label dropdown, caption di bawah gambar, markdown */
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
    [data-testid="stImageCaption"], [data-testid="stCaptionContainer"],
    figcaption, .stMarkdown p { color: #000000 !important; opacity: 1 !important; }
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
    f'<div class="header">{logo_html()}PENGOLAHAN CITRA MEDIKA IMAGE ENHANCEMENT</div>',
    unsafe_allow_html=True,
)

available = list(IMAGE_FILES)
if not available:
    found = os.listdir(IMAGE_DIR) if os.path.isdir(IMAGE_DIR) else "(folder tidak ditemukan)"
    st.error(f"Tidak ada citra yang cocok di: {IMAGE_DIR}")
    st.write("Isi folder tersebut:", found)
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

# Model histogram sama seperti di notebook (ukuran font default, plot pendek & lebar)
fig, axes = plt.subplots(1, 4, figsize=(15, 1.9))
axes[0].hist(image[:, :, 0].ravel(), bins=256, color="red", alpha=0.5, label="Red")
axes[0].hist(image[:, :, 1].ravel(), bins=256, color="green", alpha=0.5, label="Green")
axes[0].hist(image[:, :, 2].ravel(), bins=256, color="blue", alpha=0.5, label="Blue")
axes[0].set_title(name + " - Histogram RGB")
axes[0].legend()
for ax, ch, color in zip(axes[1:], range(3), ["red", "green", "blue"]):
    ax.hist(image[:, :, ch].ravel(), bins=256, color=color)
    ax.set_title(f"{name} - Channel {color.capitalize()}")
for ax in axes:
    ax.set_xlabel("Intensitas Piksel")
    ax.set_ylabel("Jumlah Piksel")
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
