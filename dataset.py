import os
from os.path import isfile, join
import logging
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from pytorch_lightning import LightningDataModule
from scipy.io import wavfile
from torch.utils.data import Dataset
from tqdm import tqdm

# Constants
MAX_FRAMES = 288
MAX_AUDIO_LEN = 184320
SEED = 1143
SAMPLING_RATE = 16000
FRAMES_PER_SECOND = 25

# Programmatic blacklist to bypass slow sharedscratch IO checking
BLACKLISTED_SAMPLES = {
    "S03581", "S03956", "S07926", "S08579", "S16057", 
    "S17163", "S20789", "S22626", "S23729"
}


def subsample_list(inp_list: List, sample_rate: float) -> List:
    random.shuffle(inp_list)
    return [inp_list[i] for i in range(int(len(inp_list) * sample_rate))]


class AVSE4Dataset(Dataset):
    def __init__(self, scenes_root, shuffle=False, seed=SEED, subsample=1,
                 clipped_batch=False, sample_items=False, test_set=False, rgb=False,
                 audio_norm=False, num_channels=1):
        super().__init__()
        assert num_channels in [1, 2], "Number of channels must be 1 or 2"
        assert os.path.isdir(scenes_root), f"Scenes root {scenes_root} not found"
        self.num_channels = num_channels
        self.mono = num_channels == 1
        self.img_size = 112
        self.audio_norm = audio_norm
        self.test_set = test_set
        self.clipped_batch = clipped_batch
        self.scenes_root = scenes_root
        self.files_list = self.build_files_list()
        
        if shuffle:
            random.seed(seed)
            random.shuffle(self.files_list)
        if subsample != 1:
            self.files_list = subsample_list(self.files_list, sample_rate=subsample)
            
        logging.info(f"Found {len(self.files_list)} utterances")
        self.rgb = rgb
        self.sample_items = sample_items

    def build_files_list(self) -> List[Tuple[str, str, str, str, str]]:
        if isinstance(self.scenes_root, list):
            return [file for root in self.scenes_root for file in self.get_files_list(root)]
        return self.get_files_list(self.scenes_root)

    def get_files_list(self, scenes_root: str) -> List[Tuple[str, str, str, str, str]]:
        files_list = []
        for file in os.listdir(scenes_root):
            if file.endswith("_target_anechoic.wav"):
                # Fast check using the extracted file ID string
                sample_id = file.split("_")[0]
                if sample_id in BLACKLISTED_SAMPLES:
                    continue

                files = (
                    join(scenes_root, file),
                    join(scenes_root, file.replace("target_anechoic", "interferer")),
                    join(scenes_root, file.replace("target_anechoic", "mono_mix")),
                    join(scenes_root, file.replace("target_anechoic.wav", "silent.mp4")),
                    join(scenes_root, file.replace("target_anechoic", "mix")),
                )
                if not self.test_set and all(isfile(f) for f in files if not f.endswith("_interferer.wav")):
                    files_list.append(files)
                elif self.test_set:
                    files_list.append(files)
        return files_list

    def __len__(self) -> int:
        return len(self.files_list)

    def __getitem__(self, idx: int) -> dict:
        max_retries = 20

        for _ in range(max_retries):
            clean_file, noise_file, noisy_mono_file, mp4_file, noisy_file = self.files_list[idx]
            selected_noisy_file = noisy_mono_file if self.mono else noisy_file

            try:
                noisy, clean, visual = self.get_data(
                    clean_file,
                    noise_file,
                    selected_noisy_file,
                    mp4_file,
                )

                data = {
                    "mixture": torch.from_numpy(noisy).float(),
                    "clean": torch.from_numpy(clean).float(),
                    "target": torch.from_numpy(clean).float(),
                    "visual": torch.from_numpy(visual).float(),
                }

                if self.sample_items:
                    data["item"] = os.path.basename(selected_noisy_file)

                return data

            except Exception as e:
                logging.error(
                    f"Error in loading data: {e}, {mp4_file}, {selected_noisy_file}"
                )
                idx = random.randint(0, len(self.files_list) - 1)

        raise RuntimeError(
            f"Failed to load a valid sample after {max_retries} retries"
        )

    @staticmethod
    def load_wav(wav_path: str, mono=False) -> np.ndarray:
        data = wavfile.read(wav_path)[1].astype(np.float32) / 32768.0
        if mono and len(data.shape) > 1:
            data = np.mean(data, axis=1)
        return data

    def get_data(self, clean_file: str, noise_file: str, noisy_file: str, mp4_file: str) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        noisy = self.load_wav(noisy_file, self.mono)
        vr = VideoReader(mp4_file, ctx=cpu(0))
        clean = self.load_wav(clean_file, self.mono) if isfile(clean_file) else np.zeros_like(noisy)

        if self.clipped_batch:
            noisy, clean, bg_frames = self.process_clipped_batch(noisy, clean, vr)
        else:
            bg_frames = self.process_full_batch(vr)

        if self.audio_norm:
            clean = clean / (np.abs(clean).max() + 1e-8)
            noisy = noisy / (np.abs(noisy).max() + 1e-8)
        if self.mono:
            clean = clean[np.newaxis, :]
            noisy = noisy[np.newaxis, :]
        else:
            clean = clean.T
            noisy = noisy.T
        return (noisy, clean,
                bg_frames if not self.rgb else bg_frames.transpose(0, 3, 1, 2))

    def process_clipped_batch(self, noisy: np.ndarray, clean: np.ndarray, vr: VideoReader) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        if clean.shape[0] > MAX_AUDIO_LEN:
            clip_idx = random.randint(0, clean.shape[0] - MAX_AUDIO_LEN)
            video_idx = int((clip_idx / SAMPLING_RATE) * FRAMES_PER_SECOND)
            clean = clean[clip_idx:clip_idx + MAX_AUDIO_LEN]
            noisy = noisy[clip_idx:clip_idx + MAX_AUDIO_LEN]
        else:
            video_idx = 0
            if self.num_channels == 2:
                clean_pad = np.zeros((MAX_AUDIO_LEN, 2))
                noisy_pad = np.zeros((MAX_AUDIO_LEN, 2))
            else:
                clean_pad = np.zeros(MAX_AUDIO_LEN)
                noisy_pad = np.zeros(MAX_AUDIO_LEN)
            clean_pad[:clean.shape[0]] = clean
            noisy_pad[:noisy.shape[0]] = noisy
            clean = clean_pad
            noisy = noisy_pad
        frames = self.get_video_frames(vr, video_idx)
        bg_frames = self.process_frames(frames)
        return noisy, clean, bg_frames

    def process_full_batch(self, vr: VideoReader) -> np.ndarray:
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()
        return self.process_frames(frames)

    def get_video_frames(self, vr: VideoReader, video_idx: int) -> np.ndarray:
        if len(vr) < MAX_FRAMES:
            return vr.get_batch(list(range(len(vr)))).asnumpy()
        max_idx = min(video_idx + MAX_FRAMES, len(vr))
        return vr.get_batch(list(range(video_idx, max_idx))).asnumpy()

    def process_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = np.array([frame[56:-56, 56:-56, :] for frame in frames])

        if not self.rgb:
            bg_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]).astype(np.float32)
        else:
            bg_frames = frames.astype(np.float32)
        bg_frames /= 255.0

        if len(bg_frames) < MAX_FRAMES:
            pad_shape = (MAX_FRAMES - len(bg_frames), self.img_size, self.img_size, 3) if self.rgb else (
                MAX_FRAMES - len(bg_frames), self.img_size, self.img_size)
            bg_frames = np.concatenate((bg_frames, np.zeros(pad_shape, dtype=bg_frames.dtype)), axis=0)

        return bg_frames[np.newaxis, ...] if not self.rgb else bg_frames


class AVSE4DataModule(LightningDataModule):
    def __init__(self, data_root, batch_size=16, audio_norm=False, rgb=True, num_channels=1):
        super().__init__()
        # Cleaned up: Removed all trailing 'validate_videos' parameters
        self.train_dataset_batch = AVSE4Dataset(join(data_root, "train/scenes"), rgb=rgb, shuffle=True,
                                                num_channels=num_channels, clipped_batch=True, sample_items=True,
                                                audio_norm=audio_norm)
        self.dev_dataset_batch = AVSE4Dataset(join(data_root, "dev/scenes"), rgb=rgb,
                                              num_channels=num_channels, clipped_batch=True,
                                              audio_norm=audio_norm)
        self.dev_dataset = AVSE4Dataset(join(data_root, "dev/scenes"), clipped_batch=True, rgb=rgb,
                                        num_channels=num_channels, sample_items=False,
                                        audio_norm=audio_norm)
        self.eval_dataset = AVSE4Dataset(join(data_root, "dev/scenes"), clipped_batch=False, rgb=rgb,
                                         num_channels=num_channels, audio_norm=audio_norm, sample_items=False,
                                         test_set=True)
        self.batch_size = batch_size

    def train_dataloader(self):
        assert len(self.train_dataset_batch) > 0, "No training data found"
        return torch.utils.data.DataLoader(self.train_dataset_batch, batch_size=self.batch_size, num_workers=4,
                                           pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        assert len(self.dev_dataset_batch) > 0, "No validation data found"
        return torch.utils.data.DataLoader(self.dev_dataset_batch, batch_size=self.batch_size, num_workers=4,
                                           pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.batch_size, num_workers=4)


if __name__ == '__main__':
    dataset = AVSE4DataModule(data_root="/home/m_gogate/data/avsec4", batch_size=1,
                              audio_norm=False, rgb=True,
                              num_channels=2).train_dataset_batch
    for i in tqdm(range(len(dataset)), ascii=True):
        data = dataset[i]
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape, v.dtype)
        break