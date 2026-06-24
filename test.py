from os.path import isfile, join
from os import makedirs
import csv

import soundfile as sf
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
import hydra

# Standard speech evaluation metrics
from pesq import pesq
from pystoi import stoi

from dataset import AVSE4DataModule

# [Fix] This imports directly from the updated model code to guarantee target 
# class definitions remain standard throughout the validation process.
from new_model import AVSE4BaselineModule
from new_core import Config

SAMPLE_RATE = 16000

# Set False to evaluate with the raw (non-averaged) training weights
# instead of the EMA-smoothed ones. 
USE_EMA_WEIGHTS = True


def compute_si_sdr(reference, distorted, max_lag=80):
    """
    Calculates Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in NumPy.
    """
    reference = reference.flatten()
    distorted = distorted.flatten()

    min_len = min(len(reference), len(distorted))
    reference = reference[:min_len]
    distorted = distorted[:min_len]

    search = min(max_lag, min_len // 4)
    best_lag = 0
    best_score = -np.inf
    for lag in range(-search, search + 1):
        if lag >= 0:
            r = reference[lag:]
            d = distorted[: len(r)]
        else:
            d = distorted[-lag:]
            r = reference[: len(d)]
        if len(r) < 10:
            continue
        denom = (np.linalg.norm(r) * np.linalg.norm(d)) + 1e-8
        score = np.dot(r, d) / denom
        if score > best_score:
            best_score = score
            best_lag = lag

    if best_lag >= 0:
        reference = reference[best_lag:]
        distorted = distorted[: len(reference)]
    else:
        distorted = distorted[-best_lag:]
        reference = reference[: len(distorted)]

    alpha = np.dot(distorted, reference) / (np.dot(reference, reference) + 1e-8)
    target = alpha * reference
    noise = distorted - target

    target_energy = np.sum(target ** 2)
    noise_energy = np.sum(noise ** 2)

    si_sdr_val = 10 * np.log10(target_energy / (noise_energy + 1e-8) + 1e-8)
    return si_sdr_val


@hydra.main(config_path="conf", config_name="eval", version_base="1.2")
def main(cfg: DictConfig):
    enhanced_root = join(cfg.save_dir, cfg.model_uid)
    makedirs(cfg.save_dir, exist_ok=True)
    makedirs(enhanced_root, exist_ok=True)

    datamodule = AVSE4DataModule(
        data_root=cfg.data.root,
        batch_size=1,
        rgb=cfg.data.rgb,
        num_channels=cfg.data.num_channels,
        audio_norm=cfg.data.audio_norm
    )

    if cfg.data.dev_set and cfg.data.eval_set:
        raise RuntimeError("Select either dev set or test set")
    elif cfg.data.dev_set:
        dataset = datamodule.dev_dataset
    elif cfg.data.eval_set:
        dataset = datamodule.eval_dataset
    else:
        raise RuntimeError("Select one of dev set and test set")

    # ==============================================================================
    # CHECKPOINT LOADING WITH ARCHITECTURE MISMATCH HANDLING
    #
    # strict=False in PyTorch/Lightning skips missing/extra keys but still
    # raises RuntimeError on shape mismatches. A checkpoint trained with an
    # older architecture (e.g. Linear d->d instead of d->2d in the transforms)
    # will crash even with strict=False. Fix: instantiate a fresh model to
    # get the current architecture's shapes, filter the checkpoint to only
    # shape-compatible keys, and load with strict=False so mismatched keys
    # fall back to random initialisation. EMA shadow is pulled directly from
    # the checkpoint dict rather than via the Lightning hook.
    # ==============================================================================
    import os

    print(f"[*] Loading checkpoint: {cfg.ckpt_path}")
    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    old_state_dict = checkpoint["state_dict"]

    # Fresh instantiation gives us the current architecture's parameter shapes
    # without attempting to load incompatible weights.
    model = AVSE4BaselineModule()
    target_state = model.state_dict()

    matched, skipped = {}, []
    for k, v in old_state_dict.items():
        if k in target_state and target_state[k].shape == v.shape:
            matched[k] = v
        else:
            ckpt_shape  = tuple(v.shape)
            model_shape = tuple(target_state[k].shape) if k in target_state else "missing"
            skipped.append(f"  {k}: ckpt {ckpt_shape} vs model {model_shape}")

    if skipped:
        print(f"[!] {len(skipped)} keys skipped (shape mismatch or not in current arch):")
        for s in skipped:
            print(s)

    load_result = model.load_state_dict(matched, strict=False)
    print(f"[*] Loaded {len(matched)} matched keys. "
          f"Random-init fallback: {len(load_result.missing_keys)} keys.")
    print(f"[SUCCESS] Model loaded from: {cfg.ckpt_path}")

    # Pull EMA shadow directly from the checkpoint dict (bypassing the
    # on_load_checkpoint Lightning hook which only fires during
    # load_from_checkpoint).
    ema_shadow = checkpoint.get("ema_shadow", None)
    if USE_EMA_WEIGHTS and ema_shadow is not None:
        with torch.no_grad():
            restored = 0
            for name, param in model.model.named_parameters():
                if name in ema_shadow:
                    param.data.copy_(ema_shadow[name].to(param.device, param.dtype))
                    restored += 1
        print(f"Found EMA shadow weights in checkpoint -- evaluating with EMA-averaged "
              f"weights ({restored} tensors restored).")
    elif USE_EMA_WEIGHTS:
        print("No EMA shadow found in this checkpoint (older checkpoint, or EMA was "
              "disabled) -- evaluating with raw weights.")
    else:
        print("USE_EMA_WEIGHTS=False -- evaluating with raw (non-averaged) weights.")

    device = torch.device("cuda:0" if not cfg.cpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    window = torch.hann_window(Config.WIN, device=device)

    csv_records = []
    csv_path = join(enhanced_root, "evaluation_report.csv")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Enhancing Speech & Evaluating"):
            raw_data = dataset[i]

            batch = {}
            for key, value in raw_data.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0).to(device)
                else:
                    batch[key] = value

            file_path = dataset.files_list[i][0]
            raw_filename = file_path.split("/")[-1]
            filename = raw_filename if raw_filename.endswith(".wav") else f"{raw_filename}.wav"
            enhanced_path = join(enhanced_root, filename)

            _, _, clean_tensor, _ = model._parse_dictionary_input(batch)

            if clean_tensor is not None:
                if clean_tensor.dim() == 3:
                    clean_tensor = clean_tensor.mean(dim=1)
                clean_audio = clean_tensor.squeeze().cpu().numpy()
            else:
                clean_audio = None

            outputs = model(batch)
            enh_spec = outputs[0]

            if enh_spec.dim() == 4:
                real_part = enh_spec[:, 0, :, :]
                imag_part = enh_spec[:, 1, :, :]
            elif enh_spec.dim() == 3:
                real_part = enh_spec[0, :, :]
                imag_part = enh_spec[1, :, :]
            else:
                raise ValueError(f"Unexpected enhanced spectrogram dimension: {enh_spec.dim()} with shape {enh_spec.shape}")

            enh_complex = torch.complex(real_part, imag_part)

            estimated_tensor = torch.istft(
                enh_complex,
                n_fft=Config.N_FFT,
                hop_length=Config.HOP,
                win_length=Config.WIN,
                window=window
            )
            estimated_audio = estimated_tensor.squeeze().cpu().numpy()

            if not isfile(enhanced_path):
                sf.write(enhanced_path, estimated_audio, samplerate=SAMPLE_RATE)

            if clean_audio is not None:
                min_len = min(len(clean_audio), len(estimated_audio))
                clean_audio_eval = clean_audio[:min_len]
                estimated_audio_eval = estimated_audio[:min_len]

                try:
                    pesq_score = pesq(SAMPLE_RATE, clean_audio_eval, estimated_audio_eval, 'wb')
                except Exception:
                    pesq_score = np.nan

                try:
                    stoi_score = stoi(clean_audio_eval, estimated_audio_eval, SAMPLE_RATE, extended=False)
                except Exception:
                    stoi_score = np.nan

                sisdr_score = compute_si_sdr(clean_audio_eval, estimated_audio_eval)
            else:
                pesq_score, stoi_score, sisdr_score = np.nan, np.nan, np.nan

            csv_records.append({
                "Filename": filename,
                "PESQ_WB": round(pesq_score, 4) if not np.isnan(pesq_score) else "NaN",
                "STOI": round(stoi_score, 4) if not np.isnan(stoi_score) else "NaN",
                "SI-SDR_dB": round(sisdr_score, 4) if not np.isnan(sisdr_score) else "NaN"
            })

    csv_columns = ["Filename", "PESQ_WB", "STOI", "SI-SDR_dB"]
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_records:
                writer.writerow(data)
        print(f"\n[SUCCESS] Performance matrix report saved directly to:\n--> {csv_path}")
    except IOError as e:
        print(f"Error compiling CSV artifact: {e}")


if __name__ == '__main__':
    main()