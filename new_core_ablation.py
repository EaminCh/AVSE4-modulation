"""
Ablation experiments on top of the base new_core.py architecture.
See abalation_readme.md for the full description of each experiment.

Run with:
    ABLATION_FREQ_WARP=true   train_ablation.py ...   # Experiment A
    ABLATION_FREQ_MOD=true    train_ablation.py ...   # Experiment B
    (both False)                                       # baseline control

new_core.py is untouched; this file subclasses only what changes.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from new_core import (
    Config as _BaseConfig,
    AVSEModel as _BaseAVSEModel,
    # re-exported unchanged so `from new_core_ablation import X` works
    # for everything new_model.py needs
    get_loss_weights,
    si_snr_loss,
    magnitude_loss,
    multi_resolution_stft_loss,
)


# ---------------------------------------------------------------------------
# Config -- adds ablation flags to the base Config
# ---------------------------------------------------------------------------

class Config(_BaseConfig):
    # Read from environment so two SLURM jobs can run simultaneously without
    # touching source files:
    #   ABLATION_FREQ_WARP=true   -> Experiment A (FrequencyWarp)
    #   ABLATION_FREQ_MOD=true    -> Experiment B (FreqAxisModulation)
    USE_FREQUENCY_WARP       = os.environ.get('ABLATION_FREQ_WARP', 'false').lower() == 'true'
    USE_FREQ_AXIS_MODULATION  = os.environ.get('ABLATION_FREQ_MOD',  'false').lower() == 'true'


# ---------------------------------------------------------------------------
# Experiment A: FrequencyWarp  (257 parameters)
# ---------------------------------------------------------------------------

class FrequencyWarp(nn.Module):
    """
    Learnable per-frequency power-law compression applied before the encoder.

    mag_warped[k] = mag[k] ^ gamma[k],   gamma[k] = sigmoid(raw_gamma[k])

    gamma < 1 compresses loud low-frequency vowels and lifts quiet
    high-frequency consonants/fricatives -- cochlea-inspired.  The warp is
    only used for encoder analysis; the mask is applied to the original
    (un-warped) spectrogram so the output signal is never distorted.

    Initialised at gamma ~= 0.85 (mild compression from step 0).
    """

    def __init__(self, n_freq: int = 257, init_gamma: float = 0.85):
        super().__init__()
        init_raw = math.log(init_gamma / (1.0 - init_gamma))
        self.raw_gamma = nn.Parameter(torch.full((n_freq,), init_raw))

    def forward(self, spec_ri: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec_ri: [B, 2, F, T]  real/imag
        Returns:
            [B, 2, F, T]  magnitude-warped, phase preserved
        """
        gamma = torch.sigmoid(self.raw_gamma)                          # [F]
        mag   = torch.sqrt(spec_ri[:, 0] ** 2 + spec_ri[:, 1] ** 2 + 1e-8)  # [B, F, T]
        mag_w = mag.pow(gamma.unsqueeze(-1))                           # [B, F, T]
        scale = mag_w / (mag + 1e-8)
        return torch.stack([spec_ri[:, 0] * scale,
                            spec_ri[:, 1] * scale], dim=1)


# ---------------------------------------------------------------------------
# Experiment B: FreqAxisModulation  (~1 920 parameters)
# ---------------------------------------------------------------------------

class FreqAxisModulation(nn.Module):
    """
    Coherence gate along the frequency axis, applied to the encoder
    bottleneck [B, C, F_enc, T] BEFORE channel flattening.

    Context C_harm comes from a depthwise conv over neighbouring frequency
    bins (harmonic neighbourhood), then gated with the same formula
    validated on the time axis:

        C_harm = depthwise_conv_over_freq(X)
        out    = X * (1 + tanh(beta * X * C_harm))

    This lets modulation distinguish speech harmonics (coherent with
    spectral neighbours) from noise (incoherent) before frequency structure
    is collapsed into the 256-dim bottleneck.
    """

    def __init__(self, channels: int = 192, freq_kernel: int = 9):
        super().__init__()
        self.freq_conv = nn.Conv1d(
            channels, channels, freq_kernel,
            padding=freq_kernel // 2, groups=channels, bias=False,
        )
        self.beta = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F_enc, T]
        Returns:
            [B, C, F_enc, T]
        """
        B, C, F_enc, T = x.shape
        # Conv1d over frequency axis: reshape to [B*T, C, F_enc]
        x_perm = x.permute(0, 3, 1, 2).reshape(B * T, C, F_enc)
        c_harm = self.freq_conv(x_perm).reshape(B, T, C, F_enc).permute(0, 2, 3, 1)
        beta   = self.beta.view(1, C, 1, 1)
        return x * (1.0 + torch.tanh(beta * x * c_harm))


# ---------------------------------------------------------------------------
# AVSEModel -- adds ablation hooks to the base model
# ---------------------------------------------------------------------------

class AVSEModel(_BaseAVSEModel):
    """
    Base AVSEModel extended with optional frequency-awareness experiments.
    All base modules are identical; only the ablation modules differ.
    """

    def __init__(
        self,
        model_type="av_modulation_v3",
        d_model=256,
        num_layers=6,
        pc_iterations=2,
        **kwargs,
    ):
        super().__init__(
            model_type=model_type,
            d_model=d_model,
            num_layers=num_layers,
            pc_iterations=pc_iterations,
            **kwargs,
        )
        if Config.USE_FREQUENCY_WARP:
            self.freq_warp = FrequencyWarp(n_freq=257)
        if Config.USE_FREQ_AXIS_MODULATION:
            self.freq_mod = FreqAxisModulation(channels=192)

    def forward(self, spec_ri, video):
        b, c, nf, t = spec_ri.shape

        # Experiment A: warp encoder input for perceptual analysis only.
        # The mask is applied to the ORIGINAL spec_ri below so the
        # enhanced output is never distorted by the warp.
        enc_input = spec_ri
        if Config.USE_FREQUENCY_WARP and hasattr(self, 'freq_warp'):
            enc_input = self.freq_warp(spec_ri)

        e1 = self.enc1(enc_input)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Experiment B: frequency-axis coherence gate before flattening.
        if Config.USE_FREQ_AXIS_MODULATION and hasattr(self, 'freq_mod'):
            e3 = self.freq_mod(e3)

        af = self.ap(e3.permute(0, 3, 1, 2).reshape(b, t, -1))

        vf = self.vfe(video).transpose(1, 2)
        vf = F.interpolate(vf, size=t, mode='linear', align_corners=False).transpose(1, 2)

        active_neurons  = []
        all_pred_errors = []

        for layer in self.layers:
            af, vf, ar, error = layer(af, vf, return_errors=self.training)
            active_neurons.append(ar)
            if error is not None:
                all_pred_errors.append(error)

        mean_active = sum(active_neurons) / len(active_neurons)

        bu = self.bu(af).reshape(b, t, 192, 33).permute(0, 2, 3, 1)

        d3_out = F.relu(self.d3(torch.cat([bu, e3], 1)))
        d3_out = self.vmod3(d3_out, vf)
        d3_out = F.interpolate(d3_out, size=(e2.shape[2], t), mode='bilinear', align_corners=False)

        d2_out = F.relu(self.d2(torch.cat([d3_out, e2], 1)))
        d2_out = self.vmod2(d2_out, vf)
        d2_out = F.interpolate(d2_out, size=(e1.shape[2], t), mode='bilinear', align_corners=False)

        d1_out = self.d1(torch.cat([d2_out, e1], 1))
        out    = F.interpolate(d1_out, size=(nf, t), mode='bilinear', align_corners=False)

        # Mask always applied to original spec_ri (not the warped enc_input)
        nr, ni = spec_ri[:, 0:1], spec_ri[:, 1:2]
        mask_r = 5.0 * torch.tanh(out[:, 0:1])
        mask_i = 5.0 * torch.tanh(out[:, 1:2])
        enh_r  = nr * mask_r - ni * mask_i
        enh_i  = nr * mask_i + ni * mask_r
        enh    = torch.cat([enh_r, enh_i], dim=1)

        vap_p = self.vap(vf)
        nce_l = self.nce(af, vf)

        if self.training:
            return enh, vap_p, nce_l, mean_active, all_pred_errors
        return enh
