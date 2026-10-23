import cv2
import numpy as np
import time

# Webcam -> Frame Acquisition -> Face Detection (Haar)
# ROI Selection (Forehead) -> Signal Extraction (RGB time series)
# Preprocessing (filtering) -> rPPG Signal
# Frequency Analysis (FFT) -> Heart Rate (BPM)

BUFFER_MAX = 300      # jumlah frame maksimum di buffer
MIN_FRAMES = 150      # minimal frame sebelum HR dihitung
LOW_HZ = 0.7          # batas bawah band jantung (42 BPM)
HIGH_HZ = 4.0         # batas atas band jantung (240 BPM)
N_FFT = 4096          # zero-padding agar resolusi frekuensi lebih halus


def load_cascade(name):
    """Load Haar Cascade (.xml): coba folder bawaan OpenCV, lalu folder kerja."""
    cascade = None
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    except AttributeError:
        pass
    if cascade is None or cascade.empty():
        cascade = cv2.CascadeClassifier(name)
    if cascade.empty():
        raise IOError(f"File cascade '{name}' tidak ditemukan / gagal dimuat")
    return cascade


def estimate_hr(signal, times):
    """
    Preprocessing + Frequency Analysis.
    Return: (bpm, sinyal rPPG hasil filter)
    """
    t = np.array(times)
    x = np.array(signal, dtype=np.float64)

    fs = (len(t) - 1) / (t[-1] - t[0])
    t_uniform = np.linspace(t[0], t[-1], len(t))
    x = np.interp(t_uniform, t, x)

    #Preprocessing
    #Detrend
    idx = np.arange(len(x))
    x = x - np.polyval(np.polyfit(idx, x, 1), idx)
    #Normalisasi
    x = (x - x.mean()) / (x.std() + 1e-8)

    #(pakai FFT mask) -> sinyal rPPG
    spec_f = np.fft.rfft(x)
    freqs_f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    spec_f[(freqs_f < LOW_HZ) | (freqs_f > HIGH_HZ)] = 0
    rppg = np.fft.irfft(spec_f, n=len(x))

    #FFT
    spectrum = np.abs(np.fft.rfft(rppg * np.hamming(len(rppg)), n=N_FFT))
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / fs)
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    peak_freq = freqs[band][np.argmax(spectrum[band])]

    return peak_freq * 60.0, rppg 


def draw_wave(frame, sig):
    """Gambar sinyal rPPG kecil di pojok kiri bawah."""
    h, w = frame.shape[:2]
    x0, y0, bw, bh = 10, h - 80, 300, 70
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (50, 50, 50), 1)
    s = np.array(sig[-bw:])
    if len(s) < 2:
        return
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    xs = np.linspace(x0, x0 + bw, len(s)).astype(np.int32)
    ys = (y0 + bh - 5 - s * (bh - 10)).astype(np.int32)
    cv2.polylines(frame, [np.stack([xs, ys], axis=1)], False, (0, 255, 255), 1)


#Load Haar Cascade
face_cascade = load_cascade('haarcascade_frontalface_default.xml')
eye_cascade = load_cascade('haarcascade_eye.xml')

#Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Kamera tidak bisa dibuka. Coba VideoCapture(1) atau tutup aplikasi lain.")

rgb_buffer = []      # RGB time series (r, g, b) per frame
time_buffer = []     # timestamp tiap frame
rppg_wave = []       # sinyal rPPG hasil filter
hr_history = []      # riwayat BPM untuk dihaluskan
hr_bpm = 0

while True:
    #Frame Acquisition
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Face Detection (Haar)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Wajah', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        #ROI Selection (Forehead)
        fh_y1 = y + int(h * 0.1)
        fh_y2 = y + int(h * 0.3)
        fh_x1 = x + int(w * 0.3)
        fh_x2 = x + int(w * 0.7)
        roi_forehead = frame[fh_y1:fh_y2, fh_x1:fh_x2]
        cv2.rectangle(frame, (fh_x1, fh_y1), (fh_x2, fh_y2), (0, 0, 255), 2)

        if roi_forehead.size > 0:
            #Signal Extraction (RGB time series)
            mean_b, mean_g, mean_r, _ = cv2.mean(roi_forehead)
            rgb_buffer.append((mean_r, mean_g, mean_b))
            time_buffer.append(time.time())
            if len(rgb_buffer) > BUFFER_MAX:
                rgb_buffer.pop(0)
                time_buffer.pop(0)

            #Preprocessing -> rPPG -> FFT -> BPM
            if len(rgb_buffer) >= MIN_FRAMES:
                green = [c[1] for c in rgb_buffer] 
                bpm, rppg_wave = estimate_hr(green, time_buffer)

                hr_history.append(bpm)
                if len(hr_history) > 10:
                    hr_history.pop(0)
                hr_bpm = float(np.mean(hr_history))

                cv2.putText(frame, f"Est. HR: {int(hr_bpm)} BPM", (x, y + h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                draw_wave(frame, rppg_wave)
            else:
                cv2.putText(frame, f"Mengumpulkan data... {len(rgb_buffer)}/{MIN_FRAMES}",
                            (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        #Eye Detection (di dalam ROI wajah)
        roi_gray = gray[y:y + int(h * 0.6), x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
            cv2.putText(frame, 'Mata', (x + ex, y + ey - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    else:

        rgb_buffer.clear()
        time_buffer.clear()
        hr_history.clear()
        rppg_wave = []
        cv2.putText(frame, "Wajah tidak terdeteksi", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Live Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
