
# -*- coding: utf-8 -*-
"""
Brain-Inspired Audio-Visual Speech Enhancement (AVSE)
=====================================================

Core principles:
  - MODULATION is the star -- no attention layers at all
  - Audio neurons and visual neurons are TWO SEPARATE POPULATIONS
  - Each population treats its own modality as receptive field (R)
    and the other modality as contextual field (C)
  - Coherence gate: output = R + R * C   (single projection, no MLP)
      * C measures cross-modal coherence
      * Coherent parts amplified, incoherent suppressed
  - Temporal context via lightweight depthwise convolution (replaces attention)
  - Friston precision: single-projection error correction
  - Competitive sparse activation with low base_k
"""

import os, tarfile, zipfile, cv2, requests, shutil, warnings, random, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import librosa, soundfile as sf
from pesq import pesq
from pystoi import stoi
from typing import Tuple, Optional, List

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

warnings.filterwarnings("ignore")




class Config:
    SEED       = 42
    SR         = 16000
    N_FFT      = 512
    HOP        = 160
    WIN        = 512
    ROOT       = "data"
    NOISE_DIR  = "noise"
    SAMPLE_DIR = "./samples"
    CKPT_DIR   = "./checkpoints"
    TRAIN_SUBS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                  16,17,18,19,20,22,23,24,25,26,27]
    TEST_SUBS  = [28,29,31,32,33,34]
    TEST_SNRS  = [-9,-6,-3,0]
    NOISE_FILES= {"BUS":"bus.wav","CAFE":"caf.wav","PED":"ped.wav","STR":"str.wav"}
    HARD_PROB  = 0.75
    TRAIN_SNRS = [-15,-12,-9,-9,-9,-9,-6,-6,-3,0]
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BS         = 64
    NUM_WORKERS= 12

    EPOCHS     = 80
    LR         = 4e-4
    LR_MIN     = 5e-5
    WARMUP_EP  = 5
    EMA_DECAY  = 0.999
    ALPHA      = 0.3
    NUM_EVAL   = 20
    QUICK_EVAL = 10
    EVAL_EVERY = 10
    EPS        = 1e-8
    D          = 256
    N_LAYERS   = 6
    W_NCE      = 0.05
    W_VAP      = 0.1
#    W_PC       = 0.05
    W_PC       = 0.005
    NCE_TEMP   = 0.07
    D_ALIGN    = 128
    PC_ITERS   = 2
    SPARSE_K   = 0.40


# =========================================================================
# CROSS-MODAL COHERENCE GATE (single projection, no MLP)
# =========================================================================

class CoherenceGate(nn.Module):
    """
    R + R * C  via single linear projection.

    Takes receptive field R and context, produces coherence gate C in [-1,1].
    Single Linear (no MLP) -- the modulation IS the computation.
    """

    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Linear(d * 2, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, R: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        C = torch.tanh(self.gate(torch.cat([R, context], dim=-1)))
        return self.norm(R + R * C)


# =========================================================================
# FRISTON PRECISION (single projections)
# =========================================================================

class FristonPrecision(nn.Module):
    """
    Lightweight Friston precision error correction.
    Single Linear for prediction, single Linear for precision.

    precision = f(error_magnitude) -- adapts per-element.
    """

    def __init__(self, d: int):
        super().__init__()
        self.predictor = nn.Linear(d, d)
        self.precision = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(
        self, x: torch.Tensor, num_iters: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_error = torch.zeros_like(x)
        for _ in range(num_iters):
            prediction = self.predictor(x)
            error = x - prediction
            
            error = F.mse_loss(x,prediction)
            #pi = torch.sigmoid(self.precision(error))
            pi = torch.sigmoid(self.precision(x) ) #added
            x = prediction + pi * error
            total_error = total_error + error.detach()
        return self.norm(x), total_error / max(num_iters, 1)


# =========================================================================
# COMPETITIVE SPARSE ACTIVATION
# =========================================================================

class CompetitiveSparse(nn.Module):
    """
    Top-k competitive inhibition. Single Linear for importance scoring.
    base_k controls the fraction of neurons that fire.
    """

    def __init__(self, d: int, base_k: float = 0.40):
        super().__init__()
        self.base_k = base_k
        self.importance = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        B, T, D = x.shape
        scores = torch.sigmoid(self.importance(x))

        k = max(1, int(self.base_k * D))
        threshold = scores.topk(k, dim=-1).values[..., -1:]
        mask = (scores >= threshold).float()

        if self.training:
            mask = mask + scores - scores.detach()

        active_ratio = mask.mean().item()
        return x * mask, active_ratio


# =========================================================================
# DUAL POPULATION MODULATION LAYER (Hierarchical Version)
# =========================================================================


class DualPopModulationLayer(nn.Module):
    """
    TEMPORAL SELF-MODULATION + CROSS-MODAL FUSION
    
    1. Temporal Self-Modulation (each modality uses its previous state):
       A_mod = A_t + A_t * A_{t-1}
       V_mod = V_t + V_t * V_{t-1}
    
    2. Cross-Modal Fusion via Modulation:
       Merged = A_mod + A_mod * V_mod
    
    Pure modulation using temporal context.
    """

    def __init__(self, d: int, pc_iters: int = 2):
        super().__init__()
        self.pc_iters = pc_iters
        
        # Transform to get current frame representation
        self.audio_transform = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d)
        )
        
        self.video_transform = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d)
        )
        
        # Context generators: extract context C from previous state
        self.audio_context_gen = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh()  # C in [-1, 1]
        )
        
        self.visual_context_gen = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh()
        )
        
        # Cross-modal context generator
        self.cross_context_gen = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh()
        )

        # Friston precision
        self.friston = FristonPrecision(d)

        # Sparse activation
        self.sparse = CompetitiveSparse(d, base_k=Config.SPARSE_K)

        # Temporal depthwise conv
        self.temporal = nn.Sequential(
            nn.Conv1d(d, d, 31, padding=15, groups=d),
            nn.BatchNorm1d(d),
            nn.SiLU()
        )
        self.temporal_norm = nn.LayerNorm(d)

        # FFN
        self.nf = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.SiLU(),
            nn.Linear(d * 2, d)
        )

        self.no = nn.LayerNorm(d)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        return_errors: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
        
        # Store previous states
        a_prev = audio  # A_{t-1}
        v_prev = video  # V_{t-1}
        
        # === 1. TEMPORAL SELF-MODULATION ===
        # Current representations
        a_curr = self.audio_transform(audio)   # A_t
        v_curr = self.video_transform(video)   # V_t
        
        # Context from PREVIOUS state
        c_audio = self.audio_context_gen(a_prev)   # C_a from A_{t-1}
        c_visual = self.visual_context_gen(v_prev)  # C_v from V_{t-1}
        
        # Temporal modulation: R_t + R_t * C_{t-1}
        a_mod = a_curr + a_curr * c_audio  # A_t + A_t * A_{t-1}
        v_mod = v_curr + v_curr * c_visual  # V_t + V_t * V_{t-1}
        
        # === 2. CROSS-MODAL FUSION ===
        # Visual modulates audio
        c_cross = self.cross_context_gen(v_mod)
        
        # Fusion: A_mod + A_mod * V_mod
        merged = a_mod + a_mod * c_cross

        # === 3. PRECISION & SPARSITY ===
        merged, pred_error = self.friston(merged, num_iters=self.pc_iters)
        merged, active_ratio = self.sparse(merged)

        # Temporal context
        t_out = self.temporal(
            self.temporal_norm(merged).transpose(1, 2)
        ).transpose(1, 2)
        merged = merged + t_out

        # FFN
        merged = merged + self.ffn(self.nf(merged))

        # === 4. UPDATE REPRESENTATIONS ===
        audio_out = self.no(audio + merged)
        video_out = video + 0.1 * v_mod

        error_out = pred_error if return_errors else None
        return audio_out, video_out, active_ratio, error_out



# =========================================================================
# VISUAL MODULATOR FOR DECODER
# =========================================================================

class CoherenceVisualModulator(nn.Module):
    """Decoder-side R + R*C modulation. Single conv for coherence."""

    def __init__(self, d_model: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(d_model, channels * 2)
        self.coh = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, vf: torch.Tensor) -> torch.Tensor:
        v = self.proj(vf).transpose(1, 2)
        gamma, beta = v.chunk(2, dim=1)
        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)

        r = x * (1.0 + gamma) + beta
        C = torch.tanh(self.coh(r))
        return r + r * C


# =========================================================================
# MAIN ARCHITECTURE
# =========================================================================

class GLU_Block(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc * 2, (5, 3), stride=(2, 1), padding=(2, 1))

    def forward(self, x):
        a, b = self.conv(x).chunk(2, 1)
        return a * torch.sigmoid(b)


class VisualFrontend(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.fe3d = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.PReLU(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
        )
        self.fe2d = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Sequential(
            nn.Linear(256, d),
            nn.LayerNorm(d)
        )

    def forward(self, x):
        x = self.fe3d(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        x = self.fe2d(x).view(B, T, -1)
        return self.proj(x)


class VAP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.p2 = nn.AvgPool1d(2, 1, 1)
        self.p4 = nn.AvgPool1d(4, 1, 2)
        self.h = nn.Sequential(nn.Linear(d * 3, d), nn.SiLU(), nn.Linear(d, 1))
        self.sm = nn.Conv1d(1, 1, 5, padding=4, bias=False)
        nn.init.constant_(self.sm.weight, 0.2)

    def forward(self, vf):
        B, T, d = vf.shape
        x = self.n(vf)
        xt = x.transpose(1, 2)
        x2 = self.p2(xt)[:, :, :T]
        x4 = self.p4(xt)[:, :, :T]
        lg = self.h(torch.cat([xt, x2, x4], 1).transpose(1, 2)).squeeze(-1)
        return torch.sigmoid(
            self.sm(F.pad(lg.unsqueeze(1), (4, 0)))[:, :, :T].squeeze(1)
        )


class InfoNCE(nn.Module):
    def __init__(self, d, da, tau):
        super().__init__()
        self.wa = nn.Linear(d, da, bias=False)
        self.wv = nn.Linear(d, da, bias=False)
        self.tau = tau

    def forward(self, a, v):
        B, T, d = a.shape
        idx = torch.arange(0, T, 4, device=a.device)
        A = F.normalize(self.wa(a[:, idx].reshape(-1, d)), dim=-1)
        V = F.normalize(self.wv(v[:, idx].reshape(-1, d)), dim=-1)
        N = A.shape[0]
        lg = (A @ V.T) / self.tau
        lb_row = torch.arange(lg.shape[0], device=A.device) % lg.shape[1]
        lb_col = torch.arange(lg.shape[1], device=A.device) % lg.shape[0]
   
        return (F.cross_entropy(lg, lb_row) + F.cross_entropy(lg.T, lb_col)) / 2.


class AVSEModel(nn.Module):
    def __init__(
        self,
        model_type="av_modulation_v3",
        d_model=256,
        num_layers=6,
        pc_iterations=2,
        **_
    ):
        super().__init__()

        # Visual frontend
        self.vfe = VisualFrontend(d_model)

        # Audio encoder
        self.enc1 = GLU_Block(2, 64)
        self.enc2 = GLU_Block(64, 128)
        self.enc3 = GLU_Block(128, 192)
        self.ap = nn.Sequential(
            nn.Linear(192 * 33, d_model),
            nn.LayerNorm(d_model)
        )

        # Dual population modulation layers (NO attention)
        self.layers = nn.ModuleList([
            DualPopModulationLayer(d=d_model, pc_iters=pc_iterations)
            for _ in range(num_layers)
        ])

        # Auxiliary tasks
        self.vap = VAP(d_model)
        self.nce = InfoNCE(d_model, Config.D_ALIGN, Config.NCE_TEMP)

        # Decoder
        self.bu = nn.Linear(d_model, 192 * 33)

        self.d3 = nn.ConvTranspose2d(
            384, 128, (5, 3), (2, 1), (2, 1), output_padding=(1, 0)
        )
        self.vmod3 = CoherenceVisualModulator(d_model, 128)

        self.d2 = nn.ConvTranspose2d(
            256, 64, (5, 3), (2, 1), (2, 1), output_padding=(1, 0)
        )
        self.vmod2 = CoherenceVisualModulator(d_model, 64)

        self.d1 = nn.ConvTranspose2d(
            128, 2, (5, 3), (2, 1), (2, 1), output_padding=(1, 0)
        )

    def forward(self, spec_ri, video):
        b, c, nf, t = spec_ri.shape

        # Encode audio
        e1 = self.enc1(spec_ri)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        af = self.ap(e3.permute(0, 3, 1, 2).reshape(b, t, -1))

        # Encode video
        vf = self.vfe(video).transpose(1, 2)
        vf = F.interpolate(
            vf, size=t, mode='linear', align_corners=False
        ).transpose(1, 2)

        # Dual population modulation (NO attention)
        active_neurons = []
        all_pred_errors = []

        for layer in self.layers:
            af, vf, ar, error = layer(
                af, vf, return_errors=self.training
            )
            active_neurons.append(ar)
            if error is not None:
                all_pred_errors.append(error)

        mean_active = sum(active_neurons) / len(active_neurons)

        # Bottleneck upsampling
        bu = self.bu(af).reshape(b, t, 192, 33).permute(0, 2, 3, 1)

        # Decoder with coherence modulation
        d3_out = F.relu(self.d3(torch.cat([bu, e3], 1)))
        d3_out = self.vmod3(d3_out, vf)
        d3_out = F.interpolate(
            d3_out, size=(e2.shape[2], t), mode='bilinear', align_corners=False
        )

        d2_out = F.relu(self.d2(torch.cat([d3_out, e2], 1)))
        d2_out = self.vmod2(d2_out, vf)
        d2_out = F.interpolate(
            d2_out, size=(e1.shape[2], t), mode='bilinear', align_corners=False
        )

        d1_out = self.d1(torch.cat([d2_out, e1], 1))
        out = F.interpolate(
            d1_out, size=(nf, t), mode='bilinear', align_corners=False
        )

        # Complex mask
        nr, ni = spec_ri[:, 0:1], spec_ri[:, 1:2]
        mask_r = 5.0 * torch.tanh(out[:, 0:1])
        mask_i = 5.0 * torch.tanh(out[:, 1:2])
        enh_r = nr * mask_r - ni * mask_i
        enh_i = nr * mask_i + ni * mask_r
        enh = torch.cat([enh_r, enh_i], dim=1)

        # Auxiliary outputs
        vap_p = self.vap(vf)
        nce_l = self.nce(af, vf)

        if self.training:
            return enh, vap_p, nce_l, mean_active, all_pred_errors
        return enh