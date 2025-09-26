import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

# -------------------
# Load WAV file
# -------------------
filename = "data_for_technical_interview_Fall2025_1678016523.170729190140.wav"
fs, data = wavfile.read(filename)

# If stereo, convert to mono
if data.ndim > 1:
    data = data.mean(axis=1)

# Normalize
data = data / np.max(np.abs(data))

# -------------------
# Function to extract a chunk of audio
# -------------------
def get_chunk(data, fs, start_min, duration_min=5):
    start_sample = int(start_min * 60 * fs)
    end_sample = int((start_min + duration_min) * 60 * fs)
    return data[start_sample:end_sample]

# -------------------
# Choose which 5-minute chunk
# Example: 0 = 0–5 min, 5 = 5–10 min, 10 = 10–15 min, etc.
# -------------------
start_min = 5   # Change this number to pick which window to analyze
chunk = get_chunk(data, fs, start_min=start_min, duration_min=5)

# Time axis for plotting
time = np.arange(len(chunk)) / fs

# -------------------
# 1. Time-domain waveform
# -------------------
plt.figure(figsize=(12, 4))
plt.plot(time, chunk, color='b')
plt.title(f"Waveform in Time Domain ({start_min}–{start_min+5} min)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------
# 2. Spectrogram
# -------------------
plt.figure(figsize=(12, 6))
f, t, Sxx = signal.spectrogram(chunk, fs, nperseg=1024, noverlap=512)
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud", cmap="viridis")
plt.colorbar(label="Power [dB]")
plt.title(f"Spectrogram ({start_min}–{start_min+5} min)")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.ylim(0, fs/2)  # Nyquist limit
plt.tight_layout()
plt.show()

# -------------------
# 3. Power Spectral Density (Welch’s method)
# -------------------
plt.figure(figsize=(8, 5))
f, Pxx = signal.welch(chunk, fs, nperseg=4096)
plt.semilogy(f, Pxx)
plt.title(f"Power Spectral Density ({start_min}–{start_min+5} min)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [V²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.show()
