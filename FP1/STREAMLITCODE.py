import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure, filters, morphology, measure, color
from scipy import ndimage as ndi
import cv2

st.set_page_config(page_title="Pengolahan Citra Lesi Kulit", layout="wide")
st.title("Pengolahan Citra Lesi Kulit")
st.write("Aplikasi sederhana untuk enhancement, filtering, histogram, evaluasi, dan segmentasi citra.")

uploaded_file = st.file_uploader(
    "Unggah citra kulit (JPG, JPEG, atau PNG)",
    type=["jpg", "jpeg", "png"]
)

def to_uint8(image):
    """Pastikan citra berada pada rentang 0–255."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1:
        image = image * 255
    return np.clip(image, 0, 255).astype(np.uint8)

def gray_blue(image):
    """Mengambil channel biru seperti eksperimen pada notebook."""
    if image.ndim == 3:
        return image[:, :, 2]
    return image

def histogram_equalization(image):
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_nonzero = cdf[cdf > 0]
    if len(cdf_nonzero) == 0 or cdf_nonzero[-1] == cdf_nonzero[0]:
        return image.copy()
    cdf_min = cdf_nonzero[0]
    lut = np.clip((cdf - cdf_min) * 255 / (cdf[-1] - cdf_min), 0, 255)
    return lut[image].astype(np.uint8)

def contrast_stretch(image):
    image_float = image.astype(float)
    low, high = image_float.min(), image_float.max()
    if high == low:
        return image.copy()
    return np.clip((image_float - low) * 255 / (high - low), 0, 255).astype(np.uint8)

def gaussian_kernel(size, sigma):
    axis = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def mse_psnr(reference, result):
    reference = reference.astype(float)
    result = result.astype(float)
    mse = np.mean((reference - result) ** 2)
    psnr = float("inf") if mse == 0 else 10 * np.log10((255**2) / mse)
    return mse, psnr

def make_histogram(image, title):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.hist(image.ravel(), bins=256, range=(0, 256))
    ax.set_title(title)
    ax.set_xlabel("Intensitas piksel")
    ax.set_ylabel("Jumlah piksel")
    ax.set_xlim(0, 255)
    fig.tight_layout()
    return fig

def _png_bytes(array):
    import io
    buffer = io.BytesIO()
    Image.fromarray(to_uint8(array)).save(buffer, format="PNG")
    return buffer.getvalue()

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_image = np.array(pil_image)
    original_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    working_image = gray_blue(rgb_image)
    working_image = to_uint8(working_image)

    st.subheader("Citra masukan")
    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_image, caption="Citra asli", use_container_width=True)
    with col2:
        st.image(working_image, caption="Channel biru yang digunakan untuk eksperimen", clamp=True, use_container_width=True)

    st.sidebar.header("Pengaturan")
    enhancement = st.sidebar.selectbox(
        "Metode enhancement",
        ["Tidak ada", "Histogram Equalization (HE)", "Adaptive Histogram Equalization (AHE)", "Contrast Stretching"]
    )
    if enhancement == "Adaptive Histogram Equalization (AHE)":
        clip_limit = st.sidebar.slider("clip_limit AHE", 0.01, 0.10, 0.03, 0.01)
    else:
        clip_limit = 0.03

    if enhancement == "Histogram Equalization (HE)":
        enhanced = histogram_equalization(working_image)
    elif enhancement == "Adaptive Histogram Equalization (AHE)":
        enhanced = to_uint8(exposure.equalize_adapthist(working_image, clip_limit=clip_limit))
    elif enhancement == "Contrast Stretching":
        enhanced = contrast_stretch(working_image)
    else:
        enhanced = working_image.copy()

    filtering = st.sidebar.selectbox("Metode filter", ["Tidak ada", "Mean", "Gaussian", "Median"])
    kernel_size = st.sidebar.select_slider("Ukuran kernel", options=[3, 5, 7], value=3)
    sigma_map = {3: 0.5, 5: 1.0, 7: 1.5}

    if filtering == "Mean":
        kernel = np.full((kernel_size, kernel_size), 1 / (kernel_size * kernel_size))
        processed = np.clip(ndi.convolve(enhanced.astype(float), kernel), 0, 255).astype(np.uint8)
    elif filtering == "Gaussian":
        kernel = gaussian_kernel(kernel_size, sigma_map[kernel_size])
        processed = np.clip(ndi.convolve(enhanced.astype(float), kernel), 0, 255).astype(np.uint8)
    elif filtering == "Median":
        processed = ndi.median_filter(enhanced, size=kernel_size)
    else:
        processed = enhanced.copy()

    st.subheader("Hasil pengolahan")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(working_image, caption="Sebelum enhancement", clamp=True, use_container_width=True)
    with c2:
        st.image(enhanced, caption=enhancement, clamp=True, use_container_width=True)
    with c3:
        st.image(processed, caption=f"Hasil akhir: {filtering}", clamp=True, use_container_width=True)

    st.subheader("Histogram")
    h1, h2 = st.columns(2)
    with h1:
        st.pyplot(make_histogram(working_image, "Histogram sebelum enhancement"))
    with h2:
        st.pyplot(make_histogram(processed, "Histogram hasil pengolahan"))

    mse, psnr = mse_psnr(working_image, processed)
    st.subheader("Evaluasi perubahan citra")
    m1, m2 = st.columns(2)
    m1.metric("MSE", f"{mse:.2f}")
    m2.metric("PSNR", "∞" if np.isinf(psnr) else f"{psnr:.2f} dB")
    st.caption("MSE dan PSNR di sini membandingkan hasil akhir dengan channel biru awal. Nilai ini mengukur perubahan terhadap referensi, bukan otomatis kualitas klinis citra.")

    st.subheader("Segmentasi sederhana dengan Otsu")
    threshold_value = filters.threshold_otsu(processed)
    mask = processed > threshold_value
    clean_mask = morphology.remove_small_objects(mask, min_size=100)
    clean_mask = morphology.remove_small_holes(clean_mask, area_threshold=100)
    seg_col1, seg_col2, seg_col3 = st.columns(3)
    with seg_col1:
        st.image(processed, caption="Citra untuk segmentasi", clamp=True, use_container_width=True)
    with seg_col2:
        st.image(mask.astype(np.uint8) * 255, caption=f"Mask Otsu (threshold {threshold_value:.1f})", clamp=True, use_container_width=True)
    with seg_col3:
        st.image(clean_mask.astype(np.uint8) * 255, caption="Mask setelah morfologi", clamp=True, use_container_width=True)

    props = measure.regionprops(clean_mask.astype(np.uint8))
    if props:
        largest = max(props, key=lambda p: p.area)
        st.write({
            "Luas area terbesar (piksel)": int(largest.area),
            "Perimeter (piksel)": round(float(largest.perimeter), 2),
            "Solidity": round(float(largest.solidity), 4),
        })
    else:
        st.warning("Tidak ditemukan area objek setelah morfologi. Coba citra lain atau periksa hasil mask.")

    st.download_button(
        "Unduh hasil akhir (PNG)",
        data=Image.fromarray(processed).tobytes() if False else _png_bytes(processed),
        file_name="hasil_pengolahan.png",
        mime="image/png"
    )
else:
    st.info("Unggah satu citra lesi kulit untuk memulai.")
    st.markdown("""
    **Fitur yang tersedia**
    - Enhancement: HE, AHE, dan Contrast Stretching
    - Filtering: Mean, Gaussian, dan Median
    - Histogram sebelum dan sesudah pengolahan
    - Evaluasi MSE dan PSNR
    - Segmentasi awal dengan threshold Otsu dan morfologi
    """)

