#Project PCM Kelompok 6 - Pengolahan Citra Medika (Streamlit)

import base64
import os
import io as bio

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import streamlit as st
from skimage import io, exposure, filters, morphology, measure

# ---------------------------------------------------------------- konfigurasi
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
FILE_NAMES = ["ISIC_0011387.jpg", "ISIC_0011157.jpg", "ISIC_0011346.jpg",
              "NV_6675.jpg", "VASC_6606.jpg", "BKL_6976.jpg"]
IMAGE_NAMES = ["ISIC_0011387 - Melanoma in situ", "ISIC_0011157 - Nevus",
               "ISIC_0011346 - Lentigo NOS", "NV_6675 - Melanocytic nevus",
               "VASC_6606 - Vascular lesion", "BKL_6976 - Benign keratosis-like lesion"]
MIN_SIZE = 300

try:                          # kompatibel dengan versi Streamlit lama dan baru
    cache = st.cache_data
except AttributeError:
    cache = st.cache(allow_output_mutation=True)

st.set_page_config(page_title="PCM Kelompok 6", layout="wide")


def logo_html():
    """Cari file logo*.png/jpg/webp di folder app ini atau images/, tampilkan di kiri header."""
    mimes = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".webp": "image/webp", ".svg": "image/svg+xml"}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for folder in (base_dir, DATA_DIR):
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(f)
            if stem.lower().startswith("logo") and ext.lower() in mimes:
                with open(os.path.join(folder, f), "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                return ('<img src="data:%s;base64,%s" '
                        'style="height:48px;position:absolute;left:20px;top:50%%;'
                        'transform:translateY(-50%%);">' % (mimes[ext.lower()], b64))
    return ""


st.markdown(
    """
    <style>
    .stApp { background-color: #ffd6d6; }
    [data-testid="stSidebar"] { background-color: #f9b8c4; }
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
    [data-testid="stMetric"] {
        background: #ffffff; border-left: 6px solid #9E182B;
        border-radius: 10px; padding: 10px 14px;
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
    /* Teks hitam: label, caption, markdown, pilihan radio */
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
    [data-testid="stRadio"] label p,
    [data-testid="stImageCaption"], [data-testid="stCaptionContainer"],
    figcaption, .stMarkdown p, [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] { color: #000000 !important; opacity: 1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title):
    st.markdown('<div class="pill">%s</div>' % title, unsafe_allow_html=True)


# ---------------------------------------------------------------- fungsi dasar
@cache
def load_images():
    return [io.imread(os.path.join(DATA_DIR, f))[:, :, :3] for f in FILE_NAMES]


def gaussian_kernel(size, sigma):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    weights = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return weights / weights.sum()


def calculate_mse(reference, result):
    return np.mean((reference.astype(float) - result.astype(float)) ** 2)


def calculate_psnr(mse):
    return float("inf") if mse == 0 else 10 * np.log10((255 ** 2) / mse)


def histogram_equalization(image):
    gray = image[:, :, 2]
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    cdf = hist.cumsum() / hist.sum()
    return np.clip(cdf[gray] * 255, 0, 255).astype(np.uint8)


def contrast_stretching(image):
    im = image[:, :, 2]
    c, d, a, b = im.min(), im.max(), 0, 255
    return ((im.astype(float) - c) * ((b - a) / (d - c)) + a).astype(np.uint8)


def apply_filter(image_cs, jenis, ukuran):
    if jenis == "Mean":
        kernel = np.full((ukuran, ukuran), 1.0 / (ukuran * ukuran))
        return np.clip(ndi.convolve(image_cs.astype(float), kernel), 0, 255).astype(np.uint8)
    if jenis == "Gaussian":
        sigma = {3: 0.5, 5: 1, 7: 1.5}[ukuran]
        return np.clip(ndi.convolve(image_cs.astype(float), gaussian_kernel(ukuran, sigma)),
                       0, 255).astype(np.uint8)
    return ndi.median_filter(image_cs, size=ukuran)


@cache
def hitung_cs_semua():
    return [contrast_stretching(im) for im in load_images()]

@cache
def hitung_ahe(i, clip):
    gray = load_images()[i][:, :, 2]
    return exposure.equalize_adapthist(gray, clip_limit=clip)
  
@cache
def hitung_metrik():
    images_cs = hitung_cs_semua()
    rows = []
    for name, image_cs in zip(IMAGE_NAMES, images_cs):
        for jenis in ["Mean", "Gaussian", "Median"]:
            for ukuran in [3, 5, 7]:
                hasil = apply_filter(image_cs, jenis, ukuran)
                mse = calculate_mse(image_cs, hasil)
                rows.append({"Image": name, "Filter": jenis, "Kernel": "%dx%d" % (ukuran, ukuran),
                             "MSE": round(mse, 3), "PSNR (dB)": round(calculate_psnr(mse), 3)})
    return pd.DataFrame(rows)


@cache
def hitung_filter_otsu():
        """Gaussian 3x3 lalu Otsu -> images_gauss, binary_images, threshold."""
    images_cs = hitung_cs_semua()
    images_gauss, binary_images, thresholds = [], [], []
    for i, image_cs in enumerate(images_cs):
               image_gauss = np.clip(ndi.convolve(image_cs.astype(float), gaussian_kernel(3, 0.5)),
        threshold = filters.threshold_otsu(image_gauss)
        images_gauss.append(image_gauss)
        binary_images.append(image_gauss < threshold)
        thresholds.append(int(threshold))
    return images_gauss, binary_images, thresholds


def largest_component(mask):
    label_image = measure.label(mask)
    return label_image == np.argmax(np.bincount(label_image.ravel())[1:]) + 1


@cache
def segmentasi(i):
    """Mengembalikan (daftar langkah, mask akhir) untuk citra ke-i, sama dengan notebook."""
    images = load_images()
    images_cs = hitung_cs_semua()
    images_gauss, binary_images, thresholds = hitung_filter_otsu()
    image, binary_image = images[i], binary_images[i]
    judul_filter = "Hasil filter (Gaussian 3x3)" 

    steps = [(image, "Citra asli (RGB)", None),
             (images_gauss[i], judul_filter, "gray"),
             (binary_image, "Otsu (T=%d)" % thresholds[i], "gray")]

    only_large = morphology.remove_small_objects(binary_image, min_size=MIN_SIZE)
    steps.append((only_large, "Hapus objek kecil", "gray"))
    mask = np.logical_not(morphology.remove_small_objects(np.logical_not(only_large), min_size=MIN_SIZE))
    steps.append((mask, "Isi lubang kecil", "gray"))

    if i == 0:       # Melanoma: Otsu bertingkat + closing + komponen terbesar
        image_gauss = images_gauss[i]
        for tahap in [2, 3]:
            t = filters.threshold_otsu(image_gauss[mask])
            mask = np.logical_and(mask, image_gauss < t)
            mask = morphology.remove_small_objects(mask, min_size=MIN_SIZE)
            steps.append((mask, "Otsu tahap %d (T=%d)" % (tahap, t), "gray"))
        mask = ndi.binary_closing(mask, iterations=15)
        mask = ndi.binary_fill_holes(mask)
        steps.append((mask, "Closing + isi lubang", "gray"))
        mask = largest_component(mask)
        steps.append((mask, "Komponen terbesar (mask akhir)", "gray"))
    elif i == 1:     # Nevus: tidak perlu tambahan
        pass
    elif i == 2:     # Lentigo: opening untuk goresan
        mask = ndi.binary_opening(mask, iterations=8)
        steps.append((mask, "Opening (hapus goresan)", "gray"))
        mask = morphology.remove_small_objects(mask, min_size=MIN_SIZE)
        steps.append((mask, "Hapus sisa objek kecil (mask akhir)", "gray"))
    else:            # NV, VASC, BKL (224x224)
        mask = ndi.binary_opening(mask, iterations=3)
        steps.append((mask, "Opening (hapus sisa garis)", "gray"))
        mask = largest_component(mask)
        steps.append((mask, "Komponen terbesar (mask akhir)", "gray"))

    steps.append((np.where(mask[:, :, np.newaxis], image, 0), "Masking", None))
    return steps, mask


def region_table():
    rows = []
    for i, name in enumerate(IMAGE_NAMES):
        _, mask = segmentasi(i)
        for p in measure.regionprops(measure.label(mask)):
            rows.append({"Image": name, "centroid-0": p.centroid[0], "centroid-1": p.centroid[1],
                         "orientation": p.orientation, "major_axis_length": p.major_axis_length,
                         "minor_axis_length": p.minor_axis_length})
    return pd.DataFrame(rows, columns=["Image", "centroid-0", "centroid-1", "orientation",
                                       "major_axis_length", "minor_axis_length"])


# ---------------------------------------------------------------- fungsi tampil
def grid(items, kolom=4, tinggi=3.2):
    """items = [(gambar, judul, cmap)] ditampilkan dalam grid matplotlib."""
    baris = (len(items) + kolom - 1) // kolom
    fig, axes = plt.subplots(baris, kolom, figsize=(4 * kolom, tinggi * baris))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (img, title, cmap) in zip(axes, items):
        if cmap == "gray" and img.dtype != bool:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        else:
            ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=9)
    plt.tight_layout()
    return fig


def tampil(fig):
    st.pyplot(fig)
    plt.close(fig)


def pilih_citra(key):
    nama = st.selectbox("Pilih citra", IMAGE_NAMES, key=key)
    return IMAGE_NAMES.index(nama)


# ---------------------------------------------------------------- cek data
ada = [os.path.exists(os.path.join(DATA_DIR, f)) for f in FILE_NAMES]
if not all(ada):
    st.error("File citra belum lengkap. Taruh file berikut di folder `%s`:" % DATA_DIR)
    for f, ok in zip(FILE_NAMES, ada):
        st.write(("OK  " if ok else "HILANG  ") + f)
    st.stop()

images = load_images()

# ---------------------------------------------------------------- tampilan
st.markdown('<div class="header">%sPENGOLAHAN CITRA MEDIKA - Kelompok 6</div>' % logo_html(),
            unsafe_allow_html=True)

HALAMAN = ["1. Citra & histogram", "2. Channel warna", "3. Enhancement", "4. Filtering",
           "5. Evaluasi MSE & PSNR", "6. Segmentasi", "7. Ekstraksi fitur"]
halaman = st.sidebar.radio("Tahap", HALAMAN)

# ---- 1
if halaman == HALAMAN[0]:
    section("Citra asli dan Histogram RGB")
    tampil(grid([(im, n, None) for im, n in zip(images, IMAGE_NAMES)], kolom=3, tinggi=3.5))
    i = pilih_citra("hist_citra")
    image = images[i]
    st.write("dtype: %s | shape: %s | min: %d | max: %d" % (image.dtype, image.shape, image.min(), image.max()))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for c, warna in enumerate(["red", "green", "blue"]):
        axes[0].hist(image[:, :, c].ravel(), bins=256, color=warna, alpha=0.4, label=warna.title())
        axes[c + 1].hist(image[:, :, c].ravel(), bins=256, color=warna)
        axes[c + 1].set_title(warna.title(), fontsize=9)
    axes[0].set_title("RGB", fontsize=9)
    axes[0].legend(fontsize=5)
    for ax in axes:
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensity Value", fontsize=8)
        ax.set_ylabel("Pixel Count", fontsize=8)
    plt.tight_layout()
    tampil(fig)

# ---- 2
elif halaman == HALAMAN[1]:
    section("Citra per channel warna")
    i = pilih_citra("channel_citra")
    image = images[i]
    items = [(image[:, :, j], n, "gray") for j, n in enumerate(["Red Channel", "Green Channel", "Blue Channel"])]
    items.append((image, "All Channel", None))
    tampil(grid(items))
    st.info("Channel Blue dipilih karena rentang intensitasnya paling lebar")

# ---- 3
elif halaman == HALAMAN[2]:
    section("Enhancement pada Channel Blue")
    i = pilih_citra("enh_citra")
    kolom = st.columns(2)
    metode = kolom[0].radio("Metode enhancement",
                            ["Histogram Equalization (HE)", "Adaptive HE (AHE)", "Contrast Stretching"])
    clip = 0.003
    if metode == "Adaptive HE (AHE)":
        clip = kolom[1].select_slider("clip_limit AHE", options=[0.001, 0.002, 0.003, 0.004, 0.005], value=0.003)

    image = images[i]
    gray = image[:, :, 2]
    if metode == "Histogram Equalization (HE)":
        hasil = histogram_equalization(image)
    elif metode == "Adaptive HE (AHE)":
        hasil = exposure.equalize_adapthist(gray, clip_limit=clip) * 255
    else:
        hasil = contrast_stretching(image)

    tampil(grid([(gray, "Channel Blue (asli)", "gray"), (hasil, metode, "gray")], kolom=2, tinggi=4.5))

    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    axes[0].hist(gray.ravel(), bins=256, range=(0, 256), color="gray")
    axes[0].set_title("Histogram Channel Blue", fontsize=9)
    axes[1].hist(hasil.ravel(), bins=256, range=(0, 256), color="gray")
    axes[1].set_title("Histogram " + metode, fontsize=9)
    for ax in axes:
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensity Value", fontsize=8)
        ax.set_ylabel("Pixel Count", fontsize=8)
    plt.tight_layout()
    tampil(fig)

# ---- 4
elif halaman == HALAMAN[3]:
    section("Filtering pada hasil Contrast Stretching")
    i = pilih_citra("fil_citra")
    image_cs = hitung_cs_semua()[i]
    kolom = st.columns(2)
    jenis = kolom[0].radio("Jenis filter", ["Mean", "Gaussian", "Median"], horizontal=True)
    ukuran = kolom[1].radio("Ukuran kernel", [3, 5, 7], horizontal=True)
    hasil = apply_filter(image_cs, jenis, ukuran)
    mse = calculate_mse(image_cs, hasil)
    tampil(grid([(image_cs, "Contrast Stretching", "gray"),
                 (hasil, "%s %dx%d" % (jenis, ukuran, ukuran), "gray")], kolom=2, tinggi=4.5))
    m1, m2 = st.columns(2)
    m1.metric("MSE", "%.3f" % mse)
    m2.metric("PSNR (dB)", "%.3f" % calculate_psnr(mse))
    with st.expander("Lihat kernel"):
        if jenis == "Median":
            st.write("Median filter tidak memakai kernel bobot (nilai tengah piksel dalam jendela %dx%d)." % (ukuran, ukuran))
        elif jenis == "Mean":
            st.write(np.full((ukuran, ukuran), 1.0 / (ukuran * ukuran)))
        else:
            st.write(gaussian_kernel(ukuran, {3: 0.5, 5: 1, 7: 1.5}[ukuran]))

# ---- 5
elif halaman == HALAMAN[4]:
    section("Evaluasi filter: MSE dan PSNR")
    df = hitung_metrik()
    pilihan = st.selectbox("Tampilkan", ["Semua citra"] + IMAGE_NAMES)
    tampil_df = df if pilihan == "Semua citra" else df[df["Image"] == pilihan]
    st.dataframe(tampil_df.reset_index(drop=True))

# ---- 6
elif halaman == HALAMAN[5]:
    section("Segmentasi dan masking")
    i = pilih_citra("seg_citra")
    steps, mask = segmentasi(i)
    tampil(grid(steps))
    st.write("Luas mask: %.1f%% dari citra" % (100.0 * mask.mean()))

# ---- 7
else:
    section("Ekstraksi fitur region (regionprops)")
    df_region = region_table()
    st.dataframe(df_region)
    try:
        buffer = bio.BytesIO()
        df_region.to_excel(buffer, index=False)
        st.download_button("Unduh hasil_region.xlsx", buffer.getvalue(), file_name="hasil_region.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.download_button("Unduh hasil_region.csv", df_region.to_csv(index=False).encode("utf-8"),
                           file_name="hasil_region.csv", mime="text/csv")
