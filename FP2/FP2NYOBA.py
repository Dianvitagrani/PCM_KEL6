import cv2
import numpy as np
import time
import csv

#PARAMETER
BUFFER_MAX = 300      
MIN_FRAMES = 150     
LOW_HZ = 0.7         
HIGH_HZ = 4.0        
N_FFT = 4096          

def load_cascade(name):
    cascade = None
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    except AttributeError:
        pass
    if cascade is None or cascade.empty():
        cascade = cv2.CascadeClassifier(name)
    if cascade.empty():
        raise IOError(f"File cascade '{name}' tidak ditemukan")
    return cascade

#PREPROCESSING
def estimate_hr(signal, times):
    t = np.array(times)
    x = np.array(signal, dtype=np.float64)

    fs = (len(t) - 1) / (t[-1] - t[0])
    t_uniform = np.linspace(t[0], t[-1], len(t))
    x = np.interp(t_uniform, t, x)

    #FILTERING
    idx = np.arange(len(x))
    x = x - np.polyval(np.polyfit(idx, x, 1), idx)
    x = (x - x.mean()) / (x.std() + 1e-8)

    spec_f = np.fft.rfft(x)
    freqs_f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    spec_f[(freqs_f < LOW_HZ) | (freqs_f > HIGH_HZ)] = 0

    #rPPG
    rppg = np.fft.irfft(spec_f, n=len(x))

    #FREQUENCY ANALYSIS (FFT)
    spectrum = np.abs(np.fft.rfft(rppg * np.hamming(len(rppg)), n=N_FFT))
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / fs)
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    peak_freq = freqs[band][np.argmax(spectrum[band])]

    return peak_freq * 60.0, rppg 

def draw_wave(frame, sig):
    h, w = frame.shape[:2]
    x0 = 10
    y0 = h - 130
    bw = w - 20
    bh = 100

    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)       
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (80, 80, 80), 1)      

    s = np.array(sig[-bw:])
    if len(s) < 2:
        return

    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    xs = np.linspace(x0, x0 + bw, len(s)).astype(np.int32)
    ys = (y0 + bh - 10 - s * (bh - 20)).astype(np.int32)

    cv2.polylines(frame, [np.stack([xs, ys], axis=1)], False, (180, 105, 255), 2)


def draw_rgb(frame, rgb_buf):
    h, w = frame.shape[:2]
    bw, bh = 300, 90                 
    x0, y0 = w - bw - 40, 80                

    cv2.rectangle(frame, (x0 - 5, y0 - 28), (x0 + bw + 5, y0 + bh + 22), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (80, 80, 80), 1)
    cv2.putText(frame, "RGB Signal over Time", (x0, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "Frame", (x0 + bw // 2 - 20, y0 + bh + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    names = ["Red", "Green", "Blue"]
    for i, (c, n) in enumerate(zip(colors, names)):
        cv2.putText(frame, n, (x0 + bw - 125 + i * 42, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

    arr = np.array(rgb_buf[-bw:], dtype=np.float64)  
    if len(arr) < 2:
        return

    means = arr.mean(axis=0)                          
    dev = arr - means                               
    gain = 14.0 / (np.abs(dev).max() + 1e-8)          

    xs = np.linspace(x0, x0 + bw, len(arr)).astype(np.int32)
    for ch, color in enumerate(colors):
        base = y0 + bh - 10 - (means[ch] / 255.0) * (bh - 20)  
        ys = np.clip(base - dev[:, ch] * gain, y0 + 2, y0 + bh - 2).astype(np.int32)
        cv2.polylines(frame, [np.stack([xs, ys], axis=1)], False, color, 2)

def fit_to_window(frame, win_name):
    _, _, win_w, win_h = cv2.getWindowImageRect(win_name)
    if win_w <= 0 or win_h <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = max(win_w / w, win_h / h)
    new_w = max(win_w, int(w * scale))
    new_h = max(win_h, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x_off = (new_w - win_w) // 2
    y_off = (new_h - win_h) // 2
    return resized[y_off:y_off + win_h, x_off:x_off + win_w]

face_cascade = load_cascade('haarcascade_frontalface_default.xml')
eye_cascade = load_cascade('haarcascade_eye_tree_eyeglasses.xml')

#WEBCAM
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise IOError("Kamera tidak bisa dibuka. Coba VideoCapture(1) atau tutup aplikasi lain.")

rgb_buffer = []      
time_buffer = []     
rppg_wave = []       
hr_history = []      
hr_bpm = 0

start_time = time.time()
csv_file = open('riwayat_bpm.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['waktu_detik', 'bpm_mentah', 'bpm_halus'])

cv2.namedWindow('Live Webcam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Webcam', 960, 540)

while True:
    #FRAME ACQUISITION
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)   # [+] mirror (tambahan)

    #FACE DETECTION (HAAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Wajah', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        #ROI SELECTION (FOREHEAD)
        fh_y1 = y + int(h * 0.1)
        fh_y2 = y + int(h * 0.3)
        fh_x1 = x + int(w * 0.3)
        fh_x2 = x + int(w * 0.7)
        roi_forehead = frame[fh_y1:fh_y2, fh_x1:fh_x2]
        cv2.rectangle(frame, (fh_x1, fh_y1), (fh_x2, fh_y2), (0, 0, 255), 2)

        if roi_forehead.size > 0:
            #SIGNAL ESTRACTION
            mean_b, mean_g, mean_r, _ = cv2.mean(roi_forehead)
            rgb_buffer.append((mean_r, mean_g, mean_b))
            time_buffer.append(time.time())
            if len(rgb_buffer) > BUFFER_MAX:
                rgb_buffer.pop(0)
                time_buffer.pop(0)

            #PREPROCESSING
            if len(rgb_buffer) >= MIN_FRAMES:
                green = [c[1] for c in rgb_buffer]  
                bpm, rppg_wave = estimate_hr(green, time_buffer)

                hr_history.append(bpm)
                if len(hr_history) > 10:
                    hr_history.pop(0)
                hr_bpm = float(np.mean(hr_history))

                csv_writer.writerow([round(time.time() - start_time, 2),
                                     round(bpm, 1), round(hr_bpm, 1)])

                cv2.putText(frame, f"Estimated HR: {int(hr_bpm)} BPM", (x, y + h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                draw_wave(frame, rppg_wave)
                draw_rgb(frame, rgb_buffer)
            else:
                cv2.putText(frame, f"Mengumpulkan data {len(rgb_buffer)}/{MIN_FRAMES}",
                            (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        #FACE DETECTION (MATA)
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

    cv2.imshow('Live Webcam', fit_to_window(frame, 'Live Webcam'))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
