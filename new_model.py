import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from pesq import pesq
from pystoi import stoi

from new_core import (
    AVSEModel, Config, si_snr_loss, magnitude_loss,
    multi_resolution_stft_loss, get_loss_weights,
)

# =========================================================================
# Summary of every fix applied across this file's history (see new_core.py
# for the parallel summary of architecture-level fixes):
#
#  [Fix A] EMAWeights class wired into Lightning hooks. Config.EMA_DECAY
#          was previously defined but never used anywhere.
#
#  [Fix B] configure_optimizers: LinearLR warmup -> CosineAnnealingLR.
#          Config.WARMUP_EP was previously defined but never used; training
#          started at full LR from epoch 0.
#
#  [Fix C] configure_gradient_clipping: norm clipping enforced (defaults
#          to 1.0, or respects trainer.gradient_clip_val if you set one).
#          Note: gradient clipping does NOT protect against NaN gradients
#          -- clip_grad_norm_ computes sqrt(sum of squares), and sqrt(nan)
#          is nan, so a NaN gradient stays NaN after "clipping". The real
#          fix for the NaN issue seen in training is in new_core.py's
#          si_snr_loss (Fix 6 there).
#
#  [Fix D] loss_pc uses the in-graph FristonPrecision errors (torch.stack
#          + mean()). A previous version called .detach() on each error,
#          which silently zeroed W_PC's gradient.
#
#  [Fix E] VAP fallback target uses per-frame energy instead of a single
#          batch-level scalar broadcast across every frame.
#
#  [Fix F] validation_step computes PESQ and STOI via ISTFT reconstruction
#          on a subset of batches. Previously imported but never called.
#
#  [Fix G] si_snr_loss added as a second, waveform-domain loss term. A
#          25-epoch run with only L1 showed L1 improve 9x while PESQ
#          improved only 17%; L1 on raw spectrograms does not correlate
#          well with perceptual quality.
#
#  [Fix H] NCE weight annealing: NCE plateaus early but at epoch 66 in one
#          run it was contributing 2.5x more gradient than the enhancement
#          loss while being completely flat. Anneal down after epoch
#          NCE_ANNEAL_START (see get_loss_weights in new_core.py).
#
#  [Fix I] W_VAP restored from 0.01 to 0.05. The lower value, applied
#          after a previous analysis, froze the VAP head at ln(2) (random
#          chance) for an entire 25-epoch run -- confirmed in real
#          training logs where vap_loss sat at 0.6939 +/- 0.0005 with no
#          measurable trend. An auxiliary loss that is given too little
#          weight does not just "contribute less" -- if its gradient is
#          already weak, lowering the weight further can extinguish
#          learning altogether.
#
#  [Fix J] magnitude_loss added: an L1 + log-L1 term computed directly on
#          spectrogram magnitude, which tracks PESQ/STOI more closely than
#          raw real/imag L1 (which penalises phase error as heavily as
#          magnitude error).
#
#  [Fix K] SI-SNR weight now ramps up over the first SISNR_WARMUP_EPOCHS
#          epochs instead of applying full weight from epoch 0, since the
#          ISTFT-based signal is noisiest while the decoder mask is still
#          far from its passthrough initialisation.
#
#  [Fix L] Defensive clamp on vap_p (and on both VAP targets) immediately
#          before every binary_cross_entropy call. This is a second line
#          of defense independent of the SI-SNR root-cause fix: any future
#          numerical issue anywhere in the model should not be able to
#          crash an 8+ hour multi-GPU job with an opaque CUDA assertion.
#
#  [Fix M] on_before_optimizer_step: a gradient-level NaN guard. A real
#          training run crashed at epoch 0 (Fix 6's bug) and, separately,
#          at epoch 37 (Fix 6b's bug, a rarer AMP-related trigger of the
#          same underlying mechanism). Rather than relying solely on
#          patching each specific mechanism as it's found, this hook
#          zeroes any non-finite gradient before the optimizer applies
#          it, so a single corrupted batch -- from this or any future,
#          still-unknown numerical edge case -- cannot crash an entire
#          multi-GPU job. Logged as metrics/nan_guard_triggered so you
#          can see exactly how often (if ever) it fires.
#
#  [Fix N] EMA persistence across checkpoint save/load. A real run showed
#          the EMA shadow was being correctly computed every training
#          step but never actually reaching disk: on_validation_epoch_end
#          restores raw weights before Lightning's ModelCheckpoint
#          callback saves (which fires on a later hook), so every saved
#          checkpoint -- including the one used for final evaluation --
#          only ever contained raw weights. Separately, on_train_start
#          always built a fresh EMA from scratch with no link to a
#          previous run, which combined with this setup's SLURM
#          auto-requeue meant the averaging window was reset to zero at
#          every requeue. on_save_checkpoint/on_load_checkpoint now
#          persist and restore the shadow so both training resumes and
#          test.py's evaluation can see it.
#
#  Note on SyncBatchNorm (multi-GPU): pass sync_batchnorm=True to
#  pl.Trainer (already added in the accompanying train.py). The temporal
#  conv block in DualPopModulationLayer uses BatchNorm1d; without sync,
#  running stats diverge across ranks under DDP.
# =========================================================================


# =========================================================================
# EMA WEIGHT HELPER  [Fix A]
# =========================================================================

class EMAWeights:
    """
    Exponential Moving Average of model parameters.

    Lifecycle (wired into Lightning hooks below):
      on_train_start            -> __init__   (snapshot initial weights)
      on_train_batch_end        -> update()   (shadow = decay*shadow + (1-decay)*param)
      on_validation_epoch_start -> apply_to()  (swap live -> EMA for eval)
      on_validation_epoch_end   -> restore_to() (swap EMA -> live for training)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay   = decay
        self.shadow  = {}
        self._backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.float(), alpha=1.0 - self.decay
                )

    def apply_to(self, model: nn.Module) -> None:
        """Replace live weights with EMA weights (call before validation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.dtype))

    def restore_to(self, model: nn.Module) -> None:
        """Restore live weights from backup (call after validation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])


# =========================================================================
# LIGHTNING MODULE
# =========================================================================

class AVSE4BaselineModule(pl.LightningModule):
    def __init__(self, num_channels: int = 2, **kwargs):
        """
        Brain-Inspired AVSE Wrapper.
        See the module-level comment block above for the full fix list.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = AVSEModel(
            model_type="av_modulation_v3",
            d_model=Config.D,
            num_layers=Config.N_LAYERS,
            pc_iterations=Config.PC_ITERS,
        )

        self.l1_loss = nn.L1Loss()

        # EMA handle, created in on_train_start so it snapshots the final
        # initialised weights (after any load_from_checkpoint).
        self.ema = None

    # ------------------------------------------------------------------
    # EMA HOOKS  [Fix A]
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        self.ema = EMAWeights(self.model, decay=Config.EMA_DECAY)

        # [Fix N] Continue averaging from a resumed checkpoint's EMA
        # state, instead of always restarting from scratch. See
        # on_load_checkpoint below for why this matters specifically for
        # this training setup (SLURM auto-requeue on walltime limits).
        loaded_shadow = getattr(self, "_loaded_ema_shadow", None)
        if loaded_shadow is not None:
            for name, val in loaded_shadow.items():
                if name in self.ema.shadow:
                    self.ema.shadow[name] = val.to(self.ema.shadow[name].device)
            print(f"Resumed EMA shadow from checkpoint ({len(loaded_shadow)} tensors).")

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.ema is not None:
            self.ema.update(self.model)

    def on_validation_epoch_start(self) -> None:
        if self.ema is not None:
            self.ema.apply_to(self.model)

    def on_validation_epoch_end(self) -> None:
        if self.ema is not None:
            self.ema.restore_to(self.model)

    # ------------------------------------------------------------------
    # EMA PERSISTENCE  [Fix N]
    # ------------------------------------------------------------------
    #
    # A real run exposed a gap: on_validation_epoch_end (above) restores
    # the raw weights before validation finishes, and Lightning's
    # ModelCheckpoint callback saves on a later hook (on_validation_end),
    # which fires AFTER that restore. So every checkpoint ever written --
    # including the one used for final evaluation -- contained only the
    # raw, non-averaged weights. The EMA shadow was computed correctly
    # every step, it just never reached disk.
    #
    # Separately, on_train_start (above) always built a fresh EMA from
    # the current weights with no link to any previous run. Combined
    # with this training setup's SLURM auto-requeue on walltime limits,
    # that meant the EMA averaging window was silently reset to zero at
    # every requeue -- only ever smoothing within a single requeue
    # segment of a few dozen epochs, never across the full run.
    #
    # Fix: stash the EMA shadow inside the checkpoint dict itself (same
    # parameter count as the model, so the size overhead is small and
    # proportional), and restore it on load so both training resumes and
    # final evaluation can see it.

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.ema is not None:
            checkpoint["ema_shadow"] = {k: v.clone() for k, v in self.ema.shadow.items()}

    def on_load_checkpoint(self, checkpoint) -> None:
        # self.ema does not exist yet at this point during a resume (it
        # is built in on_train_start, which fires later) -- and for an
        # evaluation-only load via load_from_checkpoint, there is no
        # training loop at all, so this is read directly by test.py
        # instead. Either way, stash it here; both call sites pick it
        # up afterward.
        self._loaded_ema_shadow = checkpoint.get("ema_shadow", None)

    # ------------------------------------------------------------------
    # NaN GUARD  [Fix M]
    # ------------------------------------------------------------------
    #
    # Independent of the si_snr_loss root-cause fix in new_core.py, this
    # is a second, more general line of defense: a real training run
    # crashed once at epoch 0 and once at epoch 37, with the same
    # downstream symptom but a different and rarer trigger each time.
    # Rather than only patching the specific mechanism found so far, this
    # hook makes the training loop itself resilient to ANY future source
    # of non-finite gradients, whatever it turns out to be.
    #
    # It runs after backward() (so DDP's gradient all-reduce has already
    # happened and every rank sees the same globally-reduced gradient,
    # consistent across ranks) but before the optimizer applies the
    # update. If any gradient is non-finite, every gradient is zeroed
    # before the optimizer step, which makes that step a safe no-op
    # (Adam's moving averages absorb a zero-gradient step gracefully --
    # this is a standard, well-established pattern for long, unattended
    # multi-GPU jobs) instead of crashing the entire run.

    def on_before_optimizer_step(self, optimizer) -> None:
        all_finite = True
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                all_finite = False
                break

        if not all_finite:
            self._nan_guard_count = getattr(self, "_nan_guard_count", 0) + 1
            self.log("metrics/nan_guard_triggered", 1.0,
                     on_step=True, on_epoch=True, sync_dist=True)
            self.log("metrics/nan_guard_total_count", float(self._nan_guard_count),
                     on_step=True, sync_dist=True)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        else:
            self.log("metrics/nan_guard_triggered", 0.0,
                     on_step=True, on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # INPUT UTILITIES
    # ------------------------------------------------------------------

    def _parse_dictionary_input(self, data):
        if not isinstance(data, dict):
            noisy      = data[0]
            video      = data[1]
            clean      = data[2] if len(data) > 2 else None
            vap_target = data[3] if len(data) > 3 else None
            return noisy, video, clean, vap_target

        audio_keys = ["noisy_audio", "noisy", "mix", "mixture", "audio_noisy"]
        noisy = None
        for key in audio_keys:
            if key in data:
                noisy = data[key].float() if not torch.is_complex(data[key]) else data[key]
                break

        video_keys = ["video", "visual", "video_frames", "v"]
        video = None
        for key in video_keys:
            if key in data:
                video = data[key].float()
                break

        clean_keys = ["clean", "target", "clean_audio", "s"]
        clean = None
        for key in clean_keys:
            if key in data:
                clean = data[key].float() if not torch.is_complex(data[key]) else data[key]
                break

        vap_target = data.get("vap_target", None)
        return noisy, video, clean, vap_target

    def _format_audio_to_4d(self, noisy, clean=None):
        """
        Converts any audio format into [B, 2, Freq, T] (real/imag channels).
        """
        window = torch.hann_window(Config.WIN, device=noisy.device)

        # Raw 2D waveform [B, Time]
        if noisy.dim() == 2:
            stft       = torch.stft(noisy, n_fft=Config.N_FFT, hop_length=Config.HOP,
                                    win_length=Config.WIN, window=window, return_complex=True)
            noisy_spec = torch.stack([stft.real, stft.imag], dim=1)
            clean_spec = None
            if clean is not None:
                c_stft     = torch.stft(clean, n_fft=Config.N_FFT, hop_length=Config.HOP,
                                        win_length=Config.WIN, window=window, return_complex=True)
                clean_spec = torch.stack([c_stft.real, c_stft.imag], dim=1)
            return noisy_spec, clean_spec

        # 3D tensor
        if noisy.dim() == 3:
            # Complex spectrogram [B, Freq, Time]
            if torch.is_complex(noisy):
                noisy_spec = torch.stack([noisy.real, noisy.imag], dim=1)
                clean_spec = torch.stack([clean.real, clean.imag], dim=1) if clean is not None else None
                return noisy_spec, clean_spec

            # Multi-channel waveform [B, Channels, Time]
            elif noisy.shape[1] <= 4 and noisy.shape[2] > 512:
                noisy_mono = noisy.mean(dim=1)
                stft       = torch.stft(noisy_mono, n_fft=Config.N_FFT, hop_length=Config.HOP,
                                        win_length=Config.WIN, window=window, return_complex=True)
                noisy_spec = torch.stack([stft.real, stft.imag], dim=1)
                clean_spec = None
                if clean is not None:
                    clean_mono = clean.mean(dim=1) if clean.dim() == 3 else clean
                    c_stft     = torch.stft(clean_mono, n_fft=Config.N_FFT, hop_length=Config.HOP,
                                            win_length=Config.WIN, window=window, return_complex=True)
                    clean_spec = torch.stack([c_stft.real, c_stft.imag], dim=1)
                return noisy_spec, clean_spec

            # Real spectrogram missing channel axis [B, Freq, Time]
            else:
                noisy_spec = noisy.unsqueeze(1)
                clean_spec = clean.unsqueeze(1) if clean is not None else None
                return noisy_spec, clean_spec

        # Already 4D [B, 2, Freq, Time]
        return noisy, clean

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------

    def forward(self, data):
        noisy, video, _, _ = self._parse_dictionary_input(data)
        noisy_spec, _      = self._format_audio_to_4d(noisy)
        if not self.training:
            return self.model(noisy_spec, video)
        return noisy_spec, video

    # ------------------------------------------------------------------
    # TRAINING STEP
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        noisy, video, clean, vap_target = self._parse_dictionary_input(batch)
        noisy_spec, clean_spec          = self._format_audio_to_4d(noisy, clean)

        enh, vap_p, nce_loss, mean_active, all_pred_errors = self.model(noisy_spec, video)

        # ---- Primary: spectrogram L1 ----
        loss_enh = self.l1_loss(enh, clean_spec)

        # ---- Magnitude / log-magnitude loss  [Fix J] ----
        loss_mag = magnitude_loss(enh, clean_spec)

        # ---- SI-SNR loss (waveform domain)  [Fix G, Fix 6 in new_core] ----
        window     = torch.hann_window(Config.WIN, device=enh.device)
        loss_sisnr = si_snr_loss(enh, clean_spec, window)

        # ---- Multi-resolution STFT loss  [Domain fix, round 4] ----
        # Evidenced by published AVSEC-4 work that added an STFT loss on
        # top of an SI-SDR objective as part of their leaderboard-
        # improving configuration. See multi_resolution_stft_loss's
        # docstring in new_core.py for the full reasoning.
        loss_mrstft = multi_resolution_stft_loss(enh, clean_spec, window)

        # ---- Predictive-coding error  [Fix D] ----
        if all_pred_errors:
            loss_pc = torch.stack(all_pred_errors).mean()
        else:
            loss_pc = enh.new_zeros(())

        # ---- VAP  [Fix E, Fix L] ----
        if vap_target is not None:
            tgt = vap_target.float()
            if vap_p.shape != tgt.shape:
                tgt = F.interpolate(
                    tgt.unsqueeze(1), size=vap_p.shape[-1], mode="linear"
                ).squeeze(1)
        else:
            with torch.no_grad():
                # Frame-level energy: sum over channels (dim=1) and freq
                # (dim=2) -> [B, T], one energy value per frame.
                energy    = clean_spec.pow(2).sum(dim=[1, 2])        # [B, T]
                threshold = energy.mean(dim=-1, keepdim=True) * 0.3  # [B, 1]
                tgt       = (energy > threshold).float()              # [B, T]
                if vap_p.shape != tgt.shape:
                    tgt = F.interpolate(
                        tgt.unsqueeze(1), size=vap_p.shape[-1], mode="linear"
                    ).squeeze(1)

        # Defensive clamp on both sides of the BCE call. This is a second
        # line of defense independent of the SI-SNR root-cause fix above:
        # if any future change introduces a NaN or out-of-range value
        # anywhere upstream, this stops it from crashing an entire
        # multi-GPU job with an opaque CUDA assertion several steps later.
        vap_p_safe = vap_p.clamp(min=1e-6, max=1.0 - 1e-6)
        tgt_safe   = tgt.clamp(min=0.0, max=1.0)
        loss_vap   = F.binary_cross_entropy(vap_p_safe, tgt_safe)

        # ---- Epoch-dependent loss weights  [Fix H, Fix K] ----
        w = get_loss_weights(self.current_epoch)

        total_loss = (
            loss_enh
            + w["mag"]    * loss_mag
            + w["sisnr"]  * loss_sisnr
            + w["mrstft"] * loss_mrstft
            + w["nce"]    * nce_loss
            + w["vap"]    * loss_vap
            + w["pc"]     * loss_pc
        )

        self.log("train_loss",                    total_loss,   prog_bar=True, sync_dist=True)
        self.log("metrics/enh_l1",                loss_enh,     sync_dist=True)
        self.log("metrics/mag_loss",              loss_mag,     sync_dist=True)
        self.log("metrics/sisnr_loss",            loss_sisnr,   prog_bar=True, sync_dist=True)
        self.log("metrics/mrstft_loss",           loss_mrstft,  prog_bar=True, sync_dist=True)
        self.log("metrics/nce_loss",              nce_loss,     sync_dist=True)
        self.log("metrics/nce_weight",            w["nce"],     sync_dist=True)
        self.log("metrics/sisnr_weight",          w["sisnr"],   sync_dist=True)
        self.log("metrics/vap_loss",              loss_vap,     sync_dist=True)
        self.log("metrics/pc_error",              loss_pc,      sync_dist=True)
        self.log("metrics/sparsity_firing_ratio", mean_active,  prog_bar=True, sync_dist=True)
        return total_loss

    # ------------------------------------------------------------------
    # VALIDATION STEP
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        noisy, video, clean, _ = self._parse_dictionary_input(batch)
        noisy_spec, clean_spec = self._format_audio_to_4d(noisy, clean)

        eval_dict = {"noisy": noisy_spec, "video": video}
        enh       = self(eval_dict)

        val_loss = self.l1_loss(enh, clean_spec)
        self.log("val_loss",       val_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_loss_epoch", val_loss, prog_bar=True, sync_dist=True, on_epoch=True)

        # ---- Perceptual metrics on first QUICK_EVAL batches  [Fix F] ----
        # PESQ and STOI require waveforms; reconstruct via ISTFT. Only the
        # first 4 items per batch are scored to keep validation fast.
        if batch_idx < Config.QUICK_EVAL:
            try:
                window  = torch.hann_window(Config.WIN, device=enh.device)
                enh_c   = torch.complex(enh[:, 0],        enh[:, 1])
                cln_c   = torch.complex(clean_spec[:, 0], clean_spec[:, 1])
                enh_wav = torch.istft(enh_c, n_fft=Config.N_FFT, hop_length=Config.HOP,
                                      win_length=Config.WIN, window=window)
                cln_wav = torch.istft(cln_c, n_fft=Config.N_FFT, hop_length=Config.HOP,
                                      win_length=Config.WIN, window=window)

                enh_wav = torch.nan_to_num(enh_wav, nan=0.0, posinf=0.0, neginf=0.0)
                cln_wav = torch.nan_to_num(cln_wav, nan=0.0, posinf=0.0, neginf=0.0)

                pesq_scores, stoi_scores = [], []
                n_items = min(enh_wav.shape[0], 4)
                for i in range(n_items):
                    e_np = enh_wav[i].detach().cpu().float().numpy()
                    c_np = cln_wav[i].detach().cpu().float().numpy()
                    try:
                        pesq_scores.append(pesq(Config.SR, c_np, e_np, "wb"))
                        stoi_scores.append(stoi(c_np, e_np, Config.SR, extended=False))
                    except Exception:
                        pass   # signal too short or other PESQ/STOI edge case

                if pesq_scores:
                    self.log("val_pesq", float(np.mean(pesq_scores)),
                             prog_bar=True, sync_dist=True)
                    self.log("val_stoi", float(np.mean(stoi_scores)),
                             prog_bar=True, sync_dist=True)
            except Exception:
                pass  # graceful fallback if ISTFT fails on an unusual shape

        return val_loss

    # ------------------------------------------------------------------
    # OPTIMIZER & SCHEDULERS  [Fix B]
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=Config.LR, eps=Config.EPS
        )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=Config.WARMUP_EP,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.EPOCHS - Config.WARMUP_EP,
            eta_min=Config.LR_MIN,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[Config.WARMUP_EP],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # GRADIENT CLIPPING  [Fix C]
    # ------------------------------------------------------------------

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        """
        Enforce gradient norm clipping. Respects trainer.gradient_clip_val
        if set in your Hydra config; otherwise defaults to 1.0.

        Note: this protects against large but FINITE gradient spikes from
        having five loss terms of different scales. It does NOT protect
        against NaN gradients -- clip_grad_norm_ computes the norm as
        sqrt(sum of squares), and sqrt(nan) is nan, so a NaN gradient
        passes through clipping unchanged. The NaN issue seen in training
        was fixed at its source in new_core.py's si_snr_loss instead.
        """
        clip_val = gradient_clip_val if gradient_clip_val is not None else 1.0
        self.clip_gradients(
            optimizer,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm="norm",
        )