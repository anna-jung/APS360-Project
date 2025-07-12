import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(y, sr, output_path):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_mel_spectrogram(y, sr, output_path, n_mels=128, fmax=None):
    # Compute mel spectrogram (pass audio as keyword only to match newer librosa API)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=fmax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate waveform and mel spectrogram PNGs from a WAV file.')
    parser.add_argument('input_wav', type=str, help='Path to the input WAV file')
    parser.add_argument('--waveform_png', type=str, default='waveform.png', help='Output path for waveform PNG')
    parser.add_argument('--mel_png', type=str, default='mel_spectrogram.png', help='Output path for mel spectrogram PNG')
    parser.add_argument('--n_mels', type=int, default=128, help='Number of Mel bands to generate')
    parser.add_argument('--fmax', type=float, default=None, help='Maximum frequency for Mel scale')
    args = parser.parse_args()

    # Load audio
    y, sr = librosa.load(args.input_wav, sr=None)

    # Plot and save waveform
    plot_waveform(y, sr, args.waveform_png)
    print(f"Waveform saved to {args.waveform_png}")

    # Plot and save mel spectrogram
    plot_mel_spectrogram(y=y, sr=sr, output_path=args.mel_png, n_mels=args.n_mels, fmax=args.fmax)
    print(f"Mel spectrogram saved to {args.mel_png}")

if __name__ == '__main__':
    main()
