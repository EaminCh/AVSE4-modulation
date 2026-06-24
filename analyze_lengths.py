import os
import glob
import soundfile as sf
import numpy as np

# TODO: Update this path to where your audio signals (e.g., target or mixed speech) are stored
DATA_DIR = "/users/3128393c/avse_challenge-main/avse_challenge-main/baseline/avse4/" 

def analyze_dataset_durations(base_dir):
    # Find .wav files recursively inside the audio directory
    search_path = os.path.join(base_dir, "**/*.wav")
    wav_files = glob.glob(search_path, recursive=True)
    
    if not wav_files:
        print(f"No .wav files found. Please double-check if your path is correct: {base_dir}")
        return
    
    print(f"Found {len(wav_files)} total files. Sampling durations...")
    durations = []
    
    # Scan up to 3,000 files to get a rock-solid statistical representation quickly
    sample_size = min(3000, len(wav_files))
    for i, f in enumerate(wav_files[:sample_size]):
        try:
            info = sf.info(f)
            durations.append(info.duration)
        except Exception:
            continue
            
    durations = np.array(durations)
    
    print("\n================ DATASET DURATION STATISTICS ================")
    print(f"Minimum File Length:      {durations.min():.2f} seconds")
    print(f"Average (Mean) Length:    {durations.mean():.2f} seconds")
    print(f"Median (50% of data):     {np.median(durations):.2f} seconds")
    print(f"75th Percentile:          {np.percentile(durations, 75):.2f} seconds")
    print(f"90th Percentile:          {np.percentile(durations, 90):.2f} seconds")
    print(f"95th Percentile:          {np.percentile(durations, 95):.2f} seconds")
    print(f"Maximum Sampled Length:   {durations.max():.2f} seconds")
    print("=============================================================\n")

if __name__ == "__main__":
    analyze_dataset_durations(DATA_DIR)