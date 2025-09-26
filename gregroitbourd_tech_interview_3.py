import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def main():
    # Load WAV file (your path is already set here)
    filename = "data_for_technical_interview_Fall2025_1678016523.170729190140.wav"
    fs, data = wavfile.read(filename)  # fs = sampling rate, data = waveform

    # If stereo, convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Normalize
    data = data / np.max(np.abs(data))

    # Time axis
    time = np.arange(len(data)) / fs

    while True:
        print("\nChoose a graph to display:")
        print("1 - Waveform in Time Domain")
        print("2 - Spectrogram")
        print("3 - Power Spectral Density (PSD)")
        print("4 - Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            # -------------------
            # Waveform
            # -------------------
            plt.figure(figsize=(12, 4))
            plt.plot(time, data, color='b')
            plt.title("Waveform in Time Domain")
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        elif choice == "2":
            # -------------------
            # Spectrogram
            # -------------------
            plt.figure(figsize=(12, 6))
            f, t, Sxx = signal.spectrogram(data, fs, nperseg=1024, noverlap=512)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud", cmap="viridis")
            plt.colorbar(label="Power [dB]")
            plt.title("Spectrogram")
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")
            plt.ylim(0, fs/2)
            plt.tight_layout()
            plt.show()

        elif choice == "3":
            # -------------------
            # PSD
            # -------------------
            plt.figure(figsize=(8, 5))
            f, Pxx = signal.welch(data, fs, nperseg=4096)
            plt.semilogy(f, Pxx)
            plt.title("Power Spectral Density")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("PSD [V²/Hz]")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        elif choice == "4":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
