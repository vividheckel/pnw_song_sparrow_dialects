import os
from generate_audio_file_list import list_files
from multiprocessing import Pool, cpu_count
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import warnings

audio_output = "cache/processed_spectrograms/"
plt.switch_backend('Agg')

def generate_mel_spectrogram(file, segment_length=10):
    try:
        warnings.filterwarnings("ignore")
        y, sr = librosa.load(file, sr=16000)
        y, _ = librosa.effects.trim(y)

        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        else:
            return None
        
        total_length = len(y)
        segment_samples = segment_length * sr
        num_segments = total_length // segment_samples

        id = os.path.splitext(os.path.basename(file))[0]
        tag = file.split('/')[1]
        output_dir = os.path.join(audio_output, tag, id)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(num_segments + 1):
            start_sample = i * segment_samples
            end_sample = min(start_sample + segment_samples, total_length)
            segment = y[start_sample:end_sample]

            if len(segment) > 0:
                S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)

                plt.figure(figsize=(10,4))
                librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
                plt.axis('off')

                start_time = i * segment_length
                clip_name = f"{id}_{start_time:02d}.png"
                output_file_path = os.path.join(output_dir, clip_name)
                plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
                plt.close()

        warnings.filterwarnings("default")
    except Exception as e:
        return None

def generate_spectrograms(files):
    with Pool(processes=cpu_count()) as pool:
        results = list(pool.imap(generate_mel_spectrogram, files))
    return results

if __name__ == "__main__":
    os.makedirs(audio_output, exist_ok=True)
    files = list_files()
    generate_spectrograms(files)
