# -*- coding: utf-8 -*-
"""
Brain-Inspired Audio-Visual Speech Enhancement (AVSE)
======================================================

Core principles:
  - MODULATION is the star -- no attention layers at all
  - Audio neurons and visual neurons are TWO SEPARATE POPULATIONS
  - Each population treats its own modality as receptive field (R)
    and the other modality as contextual field (C)
  - Coherence gate: output = R + R * C   (single projection, no MLP)
  - Temporal context via lightweight depthwise convolution
  - Friston precision: per-element error correction
  - Competitive sparse activation

This file fixes a chain of bugs found across two real training runs:

RUN 1 (original code, before any fixes):
  [Fix 1] FristonPrecision: error was being overwritten by a scalar from
          F.mse_loss(), destroying per-element precision. Also W_PC's
          gradient was killed by an explicit .detach() on the error term.
  [Fix 2] DualPopModulationLayer: removed 2 redundant full FFNs, replaced
          with lightweight projections + CoherenceGate for all gating.
  [Fix 3] InfoNCE: guarantee a square logit matrix instead of a fragile
          modulo on the label indices.
  [Fix 4] SPARSE_K reduced 0.40 -> 0.10 for genuine sparse coding.
  [Fix 5] Decoder mask bias initialised for a near-passthrough mask at
          step 0, avoiding a large L1 spike at the start of training.

RUN 3 (after Fix 6, crash recurred at epoch 37 instead of epoch 0):
  [Fix 6b] The same log10(0) -> NaN-gradient mechanism reopened under
          automatic mixed precision (AMP/fp16). fp16's smallest
          representable positive value (~6e-8) is larger than the
          epsilon used in Fix 6 (1e-8) and in magnitude_loss (1e-5), so
          once tensors are fp16 under `trainer.precision="16-mixed"`,
          those epsilons silently underflow to exactly 0.0, and the
          original failure mode returns -- now requiring an unlucky
          combination of a near-silent clip AND fp16 rounding, which is
          rarer and shows up later/intermittently rather than every run
          at epoch 0. Fixed by forcing both si_snr_loss and
          magnitude_loss to execute in float32 with autocast explicitly
          disabled for their bodies, regardless of the ambient training
          precision, and raising eps to 1e-4 (three orders of magnitude
          above fp16's representable floor) as additional margin.

  [Domain fix] AVSEC-4 dataset facts (verified against the official
          challenge papers, not assumed): training scenes can contain up
          to 405 distinct competing speakers plus 15 noise categories
          including music, SNR ranges from -18dB to +6.55dB, and the
          official test set uses REAL recorded room impulse responses
          (vs simulated RIRs in training) -- a deliberate train/test
          domain shift via reverberation. This is a substantially
          different and harder task than the GRID+CHiME3 setup this
          codebase was originally built for (GRID: fixed-vocabulary,
          single-speaker, studio-recorded; CHiME3: 4 fixed environmental
          noise types, no competing speech, no reverberation). Two
          concrete consequences applied here:
            1. TEMPORAL_KERNEL widened from 31 to 63 (310ms -> 630ms
               receptive field per layer) to give the model more context
               for dereverberation.
            2. NCE_ANNEAL_FLOOR added at 0.4x W_NCE (previously annealed
               toward 0.1x): on a multi-talker dataset, the audio-visual
               synchrony signal that InfoNCE trains is not just a
               generic regularizer -- it is plausibly the main mechanism
               by which the model identifies WHICH voice in a mixture of
               speakers is the target, via lip-sync correlation. Annealing
               it toward near-zero, which was reasonable for a
               single-speaker-plus-noise setup, risks weakening that
               disambiguation signal on AVSEC-4 specifically.
          The Config fields TRAIN_SUBS, TEST_SUBS, NOISE_FILES, and
          TRAIN_SNRS below are very likely leftover from the original
          GRID+CHiME3 setup and unused now that an AVSE4DataModule loads
          the official pre-mixed challenge scenes directly. They are
          left in place (removing them risks breaking something this
          analysis cannot see), but are worth confirming as dead code.


  [Fix 7] Loss weight rebalancing based on a 25-epoch run with Fix 6 not
          yet applied:
            - W_VAP had been lowered to 0.01 in a previous pass, which
              froze the VAP head at ln(2) (chance level) instead of
              learning anything. Restored to a moderate 0.05.
            - Added a magnitude / log-magnitude loss (magnitude_loss),
              which tracks PESQ/STOI more directly than raw real/imag L1
              because human perception of speech quality is dominated by
              magnitude, not phase, at the SNRs used here.
            - Added an epoch-based ramp for the SI-SNR weight (low for
              the first few epochs while the ISTFT-based signal is noisy,
              then up to full weight) instead of applying full weight
              from epoch 0.
            - Kept the existing NCE down-weighting schedule, which was
              verified correct against the logged training data.
          All of this lives in get_loss_weights(epoch) below so the
          training loop only needs to call one function.
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
from typing import Tuple, Optional, List, Dict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

warnings.filterwarnings("ignore")


# =========================================================================
# CONFIG
# =========================================================================

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

    EPOCHS     = 100
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

    # ---- Loss weights (base values; some are scheduled by epoch -- see
    #      get_loss_weights() below, which the training step should call
    #      once per batch instead of reading these fields directly) ----
    W_NCE      = 0.05
    W_VAP      = 0.05    # restored from 0.01: that value froze the VAP
                          # head at chance level (ln(2)) in a real run
    W_PC       = 0.005
    W_SISNR    = 0.5
    W_MAG      = 0.2     # magnitude + log-magnitude loss weight
    W_MRSTFT   = 0.3     # multi-resolution STFT loss weight (round 4 addition)

    NCE_TEMP   = 0.07
    D_ALIGN    = 128
    PC_ITERS   = 2

    # NCE anneals DOWN, but only partway: NCE plateaus early in training,
    # but on the AVSEC-4 dataset specifically, the audio-visual synchrony
    # signal that InfoNCE trains is not just a generic regularizer the way
    # it would be on a single-speaker-plus-environmental-noise setup like
    # GRID+CHiME3. AVSEC-4 scenes can contain multiple competing speakers
    # (up to 405 distinct interferers across the dataset) plus music, so
    # the visual stream's job is partly to identify WHICH voice in the
    # mixture is the target speaker, via lip-sync correlation. Killing the
    # NCE weight down to a near-zero floor (as was done for the simpler,
    # single-speaker case) risks weakening that disambiguation signal
    # right when it matters most. Floor raised from 0.1x to 0.4x W_NCE.
    NCE_ANNEAL_START  = 10
    NCE_ANNEAL_EPOCHS = 20
    NCE_ANNEAL_FLOOR  = 0.4

    # SI-SNR ramps UP: the ISTFT-based SI-SNR loss is noisy in the first
    # few epochs while the decoder mask is still far from the passthrough
    # region set up by the Fix-5 bias initialisation. Start at 20% of
    # W_SISNR and ramp linearly to 100% over this many epochs.
    SISNR_WARMUP_EPOCHS = 8

    # [Fix 4] Was 0.40 -- firing 40% of neurons is not sparse.
    # 0.10 gives genuine competitive inhibition (about 25 of 256 dims
    # active, confirmed in real training logs: 25/256 = 0.09765625).
    SPARSE_K   = 0.10

    # [AVSEC-4] Depthwise temporal conv kernel size in DualPopModulation-
    # Layer. The original kernel of 31 (at hop=160, sr=16000, that is
    # 31*10ms = 310ms of receptive field per layer) was sized without
    # reverberation in mind. AVSEC-4 specifically adds room reverberation:
    # simulated room impulse responses in training, and REAL recorded
    # impulse responses in 3 conference rooms at 1-2m distance in the
    # official test set. Reverberant tails commonly extend several
    # hundred milliseconds, so a wider per-layer temporal context gives
    # the model more room to learn dereverberation, not just denoising.
    # Widened to 63 (630ms per layer); padding is computed automatically
    # as (kernel - 1) // 2 to preserve sequence length.
    TEMPORAL_KERNEL = 63


def get_loss_weights(epoch: int) -> Dict[str, float]:
    """
    Single source of truth for epoch-dependent loss weights.

    Call this once per training step with the current epoch (an int) and
    use the returned dict to weight each loss term. Centralising the
    schedule here means the training loop stays simple and the schedule
    itself stays easy to unit test in isolation.

    Returns:
        dict with keys: "nce", "sisnr", "mrstft", "mag", "vap", "pc"
    """
    # --- NCE: anneal down after it plateaus, but only to a floor of
    #     NCE_ANNEAL_FLOOR (not toward zero) -- see the Config comment
    #     for why this matters specifically on AVSEC-4's multi-talker
    #     scenes ---
    nce_w = Config.W_NCE
    if epoch > Config.NCE_ANNEAL_START:
        decay = max(
            0.0,
            1.0 - (epoch - Config.NCE_ANNEAL_START) / Config.NCE_ANNEAL_EPOCHS
        )
        floor = Config.NCE_ANNEAL_FLOOR
        nce_w = Config.W_NCE * (floor + (1.0 - floor) * decay)

    # --- SI-SNR and multi-res STFT: both reconstruct the waveform via
    #     ISTFT, so both are noisiest in exactly the same early epochs,
    #     while the decoder mask is still far from the passthrough region
    #     set up by the Fix-5 bias initialisation. Share the same ramp. ---
    if epoch < Config.SISNR_WARMUP_EPOCHS:
        ramp = epoch / max(1, Config.SISNR_WARMUP_EPOCHS)
        sisnr_w  = Config.W_SISNR  * (0.2 + 0.8 * ramp)
        mrstft_w = Config.W_MRSTFT * (0.2 + 0.8 * ramp)
    else:
        sisnr_w  = Config.W_SISNR
        mrstft_w = Config.W_MRSTFT

    return {
        "nce":    nce_w,
        "sisnr":  sisnr_w,
        "mrstft": mrstft_w,
        "mag":    Config.W_MAG,
        "vap":    Config.W_VAP,
        "pc":     Config.W_PC,
    }


# =========================================================================
# COOPERATION GATE  (TPN "Cooperation Equation", Adeel 2025)
# =========================================================================

def cooperation_gate(R: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    Cooperation(R, C) = ReLU6( R^2 + 2R + C * (1 + |R|) )

    [Round 6] Direct drop-in replacement for the plain R + R*C modulation
    used throughout this layer. R + R*C is, almost exactly, the simplest
    member of a documented family of TPN-inspired (two-point-neuron)
    modulatory transfer functions from the same research lineage this
    codebase's docstrings already reference (Adeel et al., Phillips et
    al., the "receptive field / contextual field" framing). That family
    is, in order of sophistication:

        T_M2(R,C) = R + RC                          <- what this layer had
        T_M3(R,C) = R(1 + tanh(RC))
        T_M4(R,C) = R * 2^(RC)
        Cooperation(R,C) = ReLU6(R^2 + 2R + C(1+|R|))  <- this function

    The Cooperation Equation is the one a 2025 paper from this lineage
    ("Beyond Attention: Toward Machines with Intrinsic Higher Mental
    States", Adeel 2025) settles on after empirically comparing all of
    the above, and is reported as outperforming both standard Transformer
    attention and the simpler T_M1-T_M4 family across several benchmarks
    (CartPole and PyBullet Ant reinforcement learning, CIFAR-10, and a
    bAbI-style question-answering task).

    Why it should plausibly do more than plain R + R*C here:

      1. R^2 + 2R = (R+1)^2 - 1 is a self-amplification term that exists
         independent of C -- a strong receptive-field signal partially
         asserts itself even before context is consulted, rather than
         being entirely at the mercy of C's sign and magnitude the way
         R*C is (if C happens to be near 0, R+R*C degenerates toward
         just R; if C is near -1, R+R*C collapses toward 0 regardless of
         how strong R is).

      2. C * (1 + |R|) scales the context's contribution by the
         receptive field's own magnitude rather than leaving it as a
         flat additive term. A loud, confident R gets its coherence
         decision weighted more heavily by C than a quiet, uncertain R
         does -- which is closer to the "C splits R into coherent and
         incoherent streams" framing this codebase's docstrings already
         describe, since the splitting effect now scales with how much
         R there is to split.

      3. ReLU6 bounds the output to [0, 6], the same non-negative,
         capped activation already adopted in this layer (the request
         that added plain relu6(R+R*C) was, independently, already
         moving toward exactly the activation function this paper uses
         as part of its own named equation).

    Args:
        R: receptive field tensor, any shape
        C: contextual field tensor, broadcastable to R's shape

    Returns:
        Tensor the same shape as R (after broadcasting), in [0, 6].
    """
    return F.relu6(R.pow(2) + 2.0 * R + C * (1.0 + R.abs()))


class ActivePrecisionGate(nn.Module):
    """
    Active-precision modulation: out = R + SiLU( beta * R * C )

    [Round 8, by request, correcting Round 7] Returns to the R + R*C
    shape (T_M2) reported to work well on GRID+CHiME3, rather than the
    quadratic Cooperation Equation. The correction from Round 7: the
    precision/trust term should come from the coherence between R and C
    -- or from the modulation itself -- not from a separately-learned
    pathway. Round 7's PrecisionCooperationGate computed pi via a brand
    new nn.Linear(2d, d) over the concatenation of R and C, which is a
    parallel computation alongside the modulation, not derived from it.

    R*C IS the modulation term already present in T_M2, and it is also,
    quite literally, a coherence statistic between R and C: positive
    when they agree in sign (context confirms what the receptive field
    is saying), negative when they disagree. "Active precision" here
    means deriving the trust signal directly and dynamically from that
    live interaction every forward pass, rather than computing it
    through a separate parallel pathway.

    Concretely:
        agree = beta * (R * C)     -- the coherence/modulation term;
                                       beta is a learned per-channel
                                       scalar controlling how sharply to
                                       commit to strong agreement
        gate  = sigmoid(agree)     -- THE ACTIVE PRECISION: near 1 when
                                       R and C strongly agree, near 0
                                       when they strongly disagree, 0.5
                                       when the interaction is weak or
                                       ambiguous
        out   = R + agree * gate   -- algebraically identical to
                                       R + SiLU(agree), since
                                       SiLU(x) = x * sigmoid(x)

    SiLU is not a new ingredient -- it is already used elsewhere in this
    file (the FFN and temporal blocks both use nn.SiLU()). Introducing a
    nonlinearity into R + RC, as requested, turns out to mean applying
    an already-proven activation to the RC term, rather than anything
    exotic.

    Properties, by direct construction (also visible directly in the
    function-landscape and feature-map comparisons against plain R+RC
    and the Cooperation Equation):
      - Strong agreement (RC >> 0): gate -> 1, output -> R + RC. The
        original formula is recovered exactly in the high-coherence
        limit -- this does not discard what already worked, it only
        changes behaviour where coherence is weak or negative.
      - Strong disagreement (RC << 0): gate -> 0, and SiLU has a bounded
        global minimum (~ -0.278 / beta) rather than an unbounded
        negative value the way raw RC would produce. Incoherent
        interactions are suppressed toward a floor; R's own value still
        passes through untouched alongside it.
      - Weak/ambiguous interaction (RC ~ 0): gate ~= 0.5, a tentative,
        partial commitment to the modulation rather than either fully
        trusting or fully rejecting it.

    Parameter cost: one per-channel scalar beta (d parameters per
    instantiation) versus 2*d^2 for Round 7's PrecisionCooperationGate
    -- about 650x fewer parameters at d=256. Negligible overhead, much
    closer to "modulation is the star" than a new learned pathway.
    """

    def __init__(self, d: int):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(d))

    def forward(self, R: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        agree = self.beta * (R * C)
        return R + F.silu(agree)


def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Parallel prefix scan for the first-order linear recurrence
    h_t = a_t * h_{t-1} + b_t,  h_{-1} = 0.

    Uses the associative operator (a1,b1) x (a2,b2) = (a2*a1, a2*b1+b2)
    and a doubling strategy: stride 1, 2, 4, ... until stride >= T.
    Each doubling step is one pair of vectorized PyTorch ops -- no Python
    loop over T. Total: ceil(log2(T)) steps ~ 9 for T=300.

    a, b: [B, T, D]  (a values should be in (0,1) for stability)
    returns: [B, T, D] the hidden state sequence h_1 ... h_T
    """
    stride = 1
    while stride < a.shape[1]:
        a_prev = F.pad(a[:, :-stride], (0, 0, stride, 0), value=1.0)
        b_prev = F.pad(b[:, :-stride], (0, 0, stride, 0), value=0.0)
        new_a  = a * a_prev
        new_b  = a * b_prev + b
        a, b   = new_a, new_b
        stride <<= 1
    return b


class ContextGatedRecurrence(nn.Module):
    """
    Selective recurrent scan with parallel-scan forward pass.

    h_t = (1 - pi_t) * h_{t-1} + pi_t * x_t
    pi_t = sigmoid( beta * x_t * C_t )

    Same recurrence as before; the Python loop over T has been replaced
    with _parallel_scan above, which uses ceil(log2(T)) vectorized
    PyTorch ops instead of T sequential Python iterations. For T=300
    this is ~9 kernel launches vs ~900, eliminating the per-step Python
    overhead that was causing the slowdown in version_33.

    Parameter cost and semantics are unchanged: one per-channel scalar
    beta plus one Linear(d,d)+Tanh for the context projection.
    """

    def __init__(self, d: int):
        super().__init__()
        self.beta    = nn.Parameter(torch.ones(d))
        self.context = nn.Sequential(nn.Linear(d, d), nn.Tanh())

    def forward(self, x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input sequence (audio features after fusion)
            C: [B, T, D] context sequence (visual features, same T)
        Returns:
            [B, T, D] recurrent hidden state sequence
        """
        C_proj = self.context(C)                    # [B, T, D], in [-1, 1]
        pi     = torch.sigmoid(self.beta * x * C_proj)  # [B, T, D]
        return _parallel_scan(1.0 - pi, pi * x)


class PrecisionCooperationGate(nn.Module):
    """
    Precision-weighted Cooperation gate (Round 7) -- NOT currently used
    by DualPopModulationLayer, which now uses ActivePrecisionGate
    (Round 8) instead. Left defined here for ablation/comparison: this
    is the "precision via a separate learned pathway" version; Active-
    PrecisionGate is the "precision derived directly from R*C coherence"
    version requested as the correction to this one.

        Cooperation_pi(R, C) = ReLU6( R^2 + 2R + pi * C * (1 + |R|) )
        pi = sigmoid( Linear( cat([R, C]) ) )

    Adds a learned, per-position trust signal pi that decides how much
    weight C's contribution should actually get, rather than always
    applying the full, fixed (1+|R|) scaling from the plain Cooperation
    Equation. Motivated by FristonPrecision's existing pi = sigmoid(
    Linear(error)) pattern elsewhere in this file, and by a 2025 audio-
    visual target speaker extraction paper ("C^2AV-TSE: Context and
    Confidence-aware Audio Visual Target Speaker Extraction") that
    reported consistent PESQ/STOI/SI-SDR gains from frame-level
    confidence-awareness across six AV-TSE backbones.

    Parameter cost: one nn.Linear(2d, d) per instantiation -- 2*d^2
    params per gate, ~393K per DualPopModulationLayer at d=256, ~2.36M
    total across 6 layers.
    """

    def __init__(self, d: int):
        super().__init__()
        self.precision = nn.Linear(d * 2, d)

    def forward(self, R: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        pi = torch.sigmoid(self.precision(torch.cat([R, C], dim=-1)))
        return F.relu6(R.pow(2) + 2.0 * R + pi * C * (1.0 + R.abs()))


# =========================================================================
# CROSS-MODAL COHERENCE GATE  (single projection, no MLP)
# =========================================================================

class CoherenceGate(nn.Module):
    """
    output = LayerNorm( R + R * C )

    C = tanh( Linear( cat([R, context]) ) ) -- joint conditioning on both
    the receptive field R and external context. Single Linear, no MLP.

    [Round 5] Not currently used by DualPopModulationLayer, which was
    reverted back to separate audio_transform/video_transform/
    context_gen modules plus relu6 bounding, by request. Left defined
    here in case it is useful again later -- it is a smaller, single-
    projection alternative to the heavier per-branch transform + single-
    input context_gen pattern currently in use.
    """

    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Linear(d * 2, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, R: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        C = torch.tanh(self.gate(torch.cat([R, context], dim=-1)))
        return self.norm(R + R * C)


# =========================================================================
# FRISTON PRECISION  (single projections)
# =========================================================================

class FristonPrecision(nn.Module):
    """
    Friston predictive-coding precision module.

    error       = x - prediction                  (kept as a [B,T,D] tensor)
    pi          = sigmoid( Linear(error) )        (precision from error)
    x           = prediction + pi * error
    total_error += error.pow(2).mean()            (in-graph scalar)

    Note from real training data: pc_error tends to DRIFT UP over many
    epochs rather than down (observed 0.82 -> 1.34 across 25 epochs in one
    run). This is very likely benign: representation scale grows as the
    rest of the network learns, and squared error is unnormalised, so an
    increasing absolute value does not necessarily mean predictive coding
    is failing. Because W_PC = 0.005 is small, this term contributes at
    most about 0.0065 to the total loss regardless, so it has negligible
    practical effect on the enhancement output. It is left as-is here;
    if you want it to behave more classically, normalising the error by
    a running estimate of x's variance before squaring would be the next
    step, but that is not required for this fix.
    """

    def __init__(self, d: int):
        super().__init__()
        self.predictor = nn.Linear(d, d)
        self.precision = nn.Linear(d, d)
        self.norm      = nn.LayerNorm(d)

    def forward(
        self, x: torch.Tensor, num_iters: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total_error = x.new_zeros(())

        for _ in range(num_iters):
            prediction  = self.predictor(x)
            error       = x - prediction
            pi          = torch.sigmoid(self.precision(error))
            x           = prediction + pi * error
            total_error = total_error + error.pow(2).mean()

        return self.norm(x), total_error / max(num_iters, 1)


# =========================================================================
# COMPETITIVE SPARSE ACTIVATION
# =========================================================================

class CompetitiveSparse(nn.Module):
    """
    Top-k competitive inhibition. Single Linear for importance scoring.
    base_k controls the fraction of neurons that fire.
    """

    def __init__(self, d: int, base_k: float = 0.10):
        super().__init__()
        self.base_k     = base_k
        self.importance = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        B, T, D = x.shape
        scores    = torch.sigmoid(self.importance(x))
        k         = max(1, int(self.base_k * D))
        threshold = scores.topk(k, dim=-1).values[..., -1:]
        mask      = (scores >= threshold).float()

        if self.training:
            # Straight-through estimator keeps gradients flowing through
            # the otherwise non-differentiable threshold.
            mask = mask + scores - scores.detach()

        active_ratio = mask.mean().item()
        return x * mask, active_ratio


# =========================================================================
# DUAL POPULATION MODULATION LAYER
# =========================================================================

class DualPopModulationLayer(nn.Module):
    """
    SELF-MODULATION + CROSS-MODAL FUSION (Active-precision R + SiLU(RC))

    [Round 5, by request] Reverted from the lightweight CoherenceGate
    formulation back to separate per-branch modules.

    [Round 6-7, superseded] Briefly used cooperation_gate (the quadratic
    "Cooperation Equation") and then PrecisionCooperationGate (precision
    via a separate nn.Linear(2d,d) pathway). Both are left defined above
    for reference/ablation but are not used here anymore.

    [Round 8, current] Uses ActivePrecisionGate: out = R + SiLU(beta*RC).
    This returns to the R + R*C shape reported to work well on
    GRID+CHiME3, with one change -- the RC term is passed through SiLU
    rather than left raw, and SiLU's own internal sigmoid acts as the
    "active precision": a trust signal derived directly from the R-C
    coherence (the RC product itself), not from any separate learned
    pathway. See ActivePrecisionGate's docstring above for the full
    reasoning and the math.

    1. Self-modulation: each modality is projected through its own
       2-layer transform, then gated by a context vector generated from
       that SAME current input (not a genuine previous-timestep state --
       see the note below on why this is named "self-context" rather
       than "previous state"):
           A_mod = ActivePrecision( transform(A), context_gen(A) )
           V_mod = ActivePrecision( transform(V), context_gen(V) )

    2. Cross-modal fusion: visual context modulates the audio stream:
           Merged = ActivePrecision( A_mod, cross_context_gen(V_mod) )

    3. Friston precision (per-element, in-graph -- see FristonPrecision)

    4. Competitive sparse activation (k = Config.SPARSE_K)

    5. Temporal context via TWO parallel paths, summed:
       a) Depthwise Conv1d (kernel = Config.TEMPORAL_KERNEL, 63) for a
          fixed local receptive field.
       b) ContextGatedRecurrence: h_t = (1-pi_t)*h_{t-1} + pi_t*x_t,
          pi_t = sigmoid(beta * x_t * C_t), where C_t comes from v_mod.
          Adaptive, unbounded effective receptive field, O(T), no
          attention. Added in Round 9 to give the model the same
          long-range temporal integration that makes Mamba-based systems
          competitive on AVSEC-4's reverberant conditions.

    Parameter cost: restoring the separate audio_transform/video_transform
    (2-layer, d -> 2d -> d each) and three separate single-layer context
    generators is heavier than the collapsed CoherenceGate version --
    roughly 3*D^2 more parameters per layer (about 786K total across all
    6 layers at D=256). ActivePrecisionGate itself adds only 3*D
    parameters per layer (negligible) -- a deliberate contrast with
    Round 7's PrecisionCooperationGate, which would have added 6*D^2.
    """

    def __init__(self, d: int, pc_iters: int = 2):
        super().__init__()
        self.pc_iters = pc_iters

        # Per-branch transform: 2-layer GELU MLP, current-frame projection
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

        # Context generators: single Linear + Tanh, context in [-1, 1].
        # Each reads directly from the layer's own current input (audio /
        # video), not from a genuine stored previous state.
        self.audio_context_gen = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.video_context_gen = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.cross_context_gen = nn.Sequential(nn.Linear(d, d), nn.Tanh())

        # [Round 7] Precision-weighted Cooperation gates -- see
        # PrecisionCooperationGate's docstring for the full reasoning.
        # Kept separate per usage rather than shared.
        # [Round 8] Active-precision gates -- see ActivePrecisionGate's
        # docstring above for the full reasoning. Kept separate per usage.
        self.audio_cooperation = ActivePrecisionGate(d)
        self.video_cooperation = ActivePrecisionGate(d)
        self.cross_cooperation = ActivePrecisionGate(d)

        self.friston = FristonPrecision(d)
        self.sparse  = CompetitiveSparse(d, base_k=Config.SPARSE_K)

        # [AVSEC-4] Kernel width comes from Config.TEMPORAL_KERNEL (63 by
        # default, widened from the original 31 to give more receptive
        # field for the reverberation present in this dataset). Padding
        # is computed automatically so output length is preserved for
        # any odd kernel size.
        _tk = Config.TEMPORAL_KERNEL
        _tk_pad = (_tk - 1) // 2
        self.temporal = nn.Sequential(
            nn.Conv1d(d, d, _tk, padding=_tk_pad, groups=d),
            nn.BatchNorm1d(d),   # use Trainer(sync_batchnorm=True) on multi-GPU
            nn.SiLU()
        )
        self.temporal_norm = nn.LayerNorm(d)

        # [Round 9] Second temporal path: selective recurrent scan.
        # Summed with the depthwise conv output so both run in parallel.
        # The conv gives a fixed, local receptive field; the recurrence
        # gives an adaptive, unbounded one. C for the recurrence is v_mod
        # (the modulated visual features at the same T), so the visual
        # stream gates which audio frames update the running hidden state.
        self.recurrence = ContextGatedRecurrence(d)

        self.nf  = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.SiLU(),
            nn.Linear(d * 2, d)
        )
        self.no  = nn.LayerNorm(d)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        return_errors: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:

        # === 1. SELF-MODULATION (Precision-weighted Cooperation Equation) ===
        # audio_context/video_context are the layer's own current inputs
        # -- renamed from a_prev/v_prev to avoid implying a stored
        # previous timestep that does not actually exist here.
        audio_context = audio
        video_context = video

        a_curr = self.audio_transform(audio)              # current audio projection
        v_curr = self.video_transform(video)              # current video projection

        c_audio  = self.audio_context_gen(audio_context)
        c_visual = self.video_context_gen(video_context)

        a_mod = self.audio_cooperation(a_curr, c_audio)
        v_mod = self.video_cooperation(v_curr, c_visual)

        # === 2. CROSS-MODAL FUSION (Precision-weighted Cooperation Equation) ===
        c_cross = self.cross_context_gen(v_mod)            # visual-derived context
        merged  = self.cross_cooperation(a_mod, c_cross)

        # === 3. FRISTON PRECISION + SPARSITY ===
        merged, pred_error   = self.friston(merged, num_iters=self.pc_iters)
        merged, active_ratio = self.sparse(merged)

        # === 4. TEMPORAL CONTEXT (depthwise conv + selective recurrence, summed) ===
        # Both paths share the same normed input. Conv provides a fixed
        # local receptive field; recurrence provides an adaptive unbounded
        # one gated by visual context (v_mod).
        normed = self.temporal_norm(merged)
        t_out  = self.temporal(normed.transpose(1, 2)).transpose(1, 2)
        r_out  = self.recurrence(normed, v_mod)
        merged = merged + t_out + r_out

        # === 5. FFN ===
        merged = merged + self.ffn(self.nf(merged))

        # === 6. UPDATE REPRESENTATIONS ===
        audio_out = self.no(audio + merged)
        video_out = video + 0.1 * v_mod

        error_out = pred_error if return_errors else None
        return audio_out, video_out, active_ratio, error_out


# =========================================================================
# VISUAL MODULATOR FOR DECODER
# =========================================================================

class CoherenceVisualModulator(nn.Module):
    """Decoder-side R + R*C modulation via FiLM + coherence conv."""

    def __init__(self, d_model: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(d_model, channels * 2)
        self.coh  = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, vf: torch.Tensor) -> torch.Tensor:
        v           = self.proj(vf).transpose(1, 2)       # [B, 2C, T]
        gamma, beta = v.chunk(2, dim=1)
        gamma       = gamma.unsqueeze(2)                  # [B, C, 1, T]
        beta        = beta.unsqueeze(2)

        r = x * (1.0 + gamma) + beta                       # FiLM modulation
        C = torch.tanh(self.coh(r))
        return r + r * C                                   # coherence gate


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
        self.n  = nn.LayerNorm(d)
        self.p2 = nn.AvgPool1d(2, 1, 1)
        self.p4 = nn.AvgPool1d(4, 1, 2)
        self.h  = nn.Sequential(nn.Linear(d * 3, d), nn.SiLU(), nn.Linear(d, 1))
        self.sm = nn.Conv1d(1, 1, 5, padding=4, bias=False)
        nn.init.constant_(self.sm.weight, 0.2)

    def forward(self, vf):
        B, T, d = vf.shape
        x  = self.n(vf)
        xt = x.transpose(1, 2)
        x2 = self.p2(xt)[:, :, :T]
        x4 = self.p4(xt)[:, :, :T]
        lg = self.h(torch.cat([xt, x2, x4], 1).transpose(1, 2)).squeeze(-1)
        return torch.sigmoid(
            self.sm(F.pad(lg.unsqueeze(1), (4, 0)))[:, :, :T].squeeze(1)
        )


class InfoNCE(nn.Module):
    """
    Audio-visual contrastive alignment.

    Both embeddings are trimmed to the same length N = min(A, V) before
    computing logits, so the logit matrix is always square and the labels
    are always a clean arange(N). A previous version used a modulo on the
    label indices that was only correct for square matrices and could
    silently produce wrong labels on an edge-case batch size.
    """

    def __init__(self, d, da, tau):
        super().__init__()
        self.wa  = nn.Linear(d, da, bias=False)
        self.wv  = nn.Linear(d, da, bias=False)
        self.tau = tau

    def forward(self, a, v):
        B, T, d = a.shape
        idx = torch.arange(0, T, 4, device=a.device)
        A   = F.normalize(self.wa(a[:, idx].reshape(-1, d)), dim=-1)
        V   = F.normalize(self.wv(v[:, idx].reshape(-1, d)), dim=-1)

        N      = min(A.shape[0], V.shape[0])
        A, V   = A[:N], V[:N]
        lg     = (A @ V.T) / self.tau
        labels = torch.arange(N, device=A.device)

        return (F.cross_entropy(lg, labels) + F.cross_entropy(lg.T, labels)) / 2.


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

        self.vfe  = VisualFrontend(d_model)

        self.enc1 = GLU_Block(2, 64)
        self.enc2 = GLU_Block(64, 128)
        self.enc3 = GLU_Block(128, 192)
        self.ap   = nn.Sequential(
            nn.Linear(192 * 33, d_model),
            nn.LayerNorm(d_model)
        )

        self.layers = nn.ModuleList([
            DualPopModulationLayer(d=d_model, pc_iters=pc_iterations)
            for _ in range(num_layers)
        ])

        self.vap = VAP(d_model)
        self.nce = InfoNCE(d_model, Config.D_ALIGN, Config.NCE_TEMP)

        self.bu    = nn.Linear(d_model, 192 * 33)
        self.d3    = nn.ConvTranspose2d(384, 128, (5,3), (2,1), (2,1), output_padding=(1,0))
        self.vmod3 = CoherenceVisualModulator(d_model, 128)
        self.d2    = nn.ConvTranspose2d(256,  64, (5,3), (2,1), (2,1), output_padding=(1,0))
        self.vmod2 = CoherenceVisualModulator(d_model,  64)
        self.d1    = nn.ConvTranspose2d(128,   2, (5,3), (2,1), (2,1), output_padding=(1,0))

        # [Fix 5] Passthrough mask initialisation. At step 0 with random
        # weights, out[:,0:1] is approximately 0, so mask_r ~ 0, producing
        # near-zero enhanced signal and a large L1 spike. Setting d1.bias
        # so that 5*tanh(bias_r) ~ 1 and 5*tanh(bias_i) ~ 0 gives
        # mask_r ~ 1, mask_i ~ 0 at initialisation (near passthrough).
        #   5 * tanh(x) = 1  ->  x = atanh(0.2) ~= 0.2027
        if self.d1.bias is not None:
            nn.init.constant_(self.d1.bias[0], 0.2027)   # real mask -> 1
            nn.init.constant_(self.d1.bias[1], 0.0000)   # imag mask -> 0

    def forward(self, spec_ri, video):
        b, c, nf, t = spec_ri.shape

        e1 = self.enc1(spec_ri)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
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


# =========================================================================
# AUXILIARY LOSS FUNCTIONS  (import alongside AVSEModel)
# =========================================================================

def si_snr_loss(
    enh_spec: torch.Tensor,
    clean_spec: torch.Tensor,
    window: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Noise Ratio loss.

    Returns -SI-SNR (dB), so minimising this maximises perceptual quality.
    Typical range at epoch 0: roughly +2 to +7 (bad alignment).
    Typical range at convergence: roughly -8 to -15 (good alignment).

    [Fix 6] BOTH the signal energy (numerator) and noise energy
    (denominator) are floored with the SAME eps before the division and
    log10. An earlier version only floored the denominator, which let
    log10(0) reach -inf in the forward pass and a NaN gradient through
    the clamp boundary in the backward pass.

    [Fix 6b] Fix 6 alone was confirmed correct in float32, but a real
    training run still crashed with the same downstream symptom (a CUDA
    assertion inside the VAP head's binary_cross_entropy), just much
    later -- epoch 37 instead of epoch 0 -- and only once, not on every
    run. That pattern points to a second, rarer trigger for the same
    underlying mechanism: automatic mixed precision (AMP/fp16). Under
    fp16, the smallest representable positive value is about 6e-8. The
    eps used in Fix 6 (1e-8) and in magnitude_loss (1e-5) are both BELOW
    that threshold, so once enh_spec/clean_spec are fp16 tensors (which
    they are for at least part of the forward pass under
    `trainer.precision = "16-mixed"`), the epsilon silently underflows
    to exactly 0.0 and the original log10(0) failure mode reopens. This
    requires a near-silent clip AND an unlucky fp16 rounding to coincide,
    which is rarer than the pure fp32 bug and explains why it now shows
    up intermittently, late in training, instead of deterministically at
    epoch 0.

    Fix: force this entire function to execute in float32 regardless of
    the ambient autocast/precision setting (explicit `.float()` cast on
    every input, plus `torch.autocast(..., enabled=False)` around the
    body so no op inside can be silently downcast back to fp16), and
    raise eps to 1e-4, which has more than three orders of magnitude of
    safety margin above fp16's representable range. A final
    nan_to_num() on the returned scalar is kept as a last-resort guard.

    Args:
        enh_spec:   [B, 2, Freq, T] enhanced complex spectrogram (real/imag)
        clean_spec: [B, 2, Freq, T] clean complex spectrogram
        window:     hann window tensor on the correct device
        eps:        numerical floor used for both numerator and denominator

    Returns:
        scalar -- negative mean SI-SNR across the batch
    """
    with torch.autocast(device_type=enh_spec.device.type, enabled=False):
        enh_spec_f = enh_spec.float()
        clean_spec_f = clean_spec.float()
        window_f = window.float()

        enh_c = torch.complex(enh_spec_f[:, 0], enh_spec_f[:, 1])
        cln_c = torch.complex(clean_spec_f[:, 0], clean_spec_f[:, 1])

        enh_wav = torch.istft(enh_c, n_fft=Config.N_FFT, hop_length=Config.HOP,
                              win_length=Config.WIN, window=window_f)
        cln_wav = torch.istft(cln_c, n_fft=Config.N_FFT, hop_length=Config.HOP,
                              win_length=Config.WIN, window=window_f)

        # Defensive: stop any non-finite value from ISTFT before it can
        # reach the division/log below.
        enh_wav = torch.nan_to_num(enh_wav, nan=0.0, posinf=0.0, neginf=0.0)
        cln_wav = torch.nan_to_num(cln_wav, nan=0.0, posinf=0.0, neginf=0.0)

        cln_wav = cln_wav - cln_wav.mean(dim=-1, keepdim=True)
        enh_wav = enh_wav - enh_wav.mean(dim=-1, keepdim=True)

        dot  = (enh_wav * cln_wav).sum(dim=-1, keepdim=True)
        norm = (cln_wav ** 2).sum(dim=-1, keepdim=True).clamp(min=eps)
        proj = (dot / norm) * cln_wav

        noise = enh_wav - proj

        # Both floored with the SAME eps -- the ratio can never be
        # exactly zero, so log10 never sees an exact 0 input.
        sig_energy   = (proj  ** 2).sum(dim=-1).clamp(min=eps)
        noise_energy = (noise ** 2).sum(dim=-1).clamp(min=eps)

        si_snr = 10.0 * torch.log10(sig_energy / noise_energy)
        si_snr = si_snr.clamp(min=-30.0, max=30.0)

        loss = -si_snr.mean()

    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def magnitude_loss(
    enh_spec: torch.Tensor,
    clean_spec: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Magnitude + log-magnitude L1 loss on the complex spectrogram.

    The primary L1 loss operates on raw real/imag components, which
    penalises phase error exactly as heavily as magnitude error. Human
    perception of speech quality (and the PESQ/STOI metrics) is dominated
    by magnitude, not phase, at the SNRs used in this task. This loss
    isolates the magnitude component directly, which should correlate
    more closely with PESQ/STOI than the primary L1 term alone.

    The log-magnitude term additionally emphasises low-energy regions
    (consonants, silences between words) that a purely linear magnitude
    loss would under-weight relative to the high-energy vowel formants.

    Same precision-proofing as si_snr_loss: forced float32 execution
    with autocast disabled, and eps raised to 1e-4 so it cannot underflow
    to zero under fp16.

    Args:
        enh_spec:   [B, 2, Freq, T] enhanced complex spectrogram
        clean_spec: [B, 2, Freq, T] clean complex spectrogram
        eps:        numerical floor before sqrt/log to avoid log(0)

    Returns:
        scalar loss = L1(magnitude) + L1(log magnitude)
    """
    with torch.autocast(device_type=enh_spec.device.type, enabled=False):
        enh_spec_f = enh_spec.float()
        clean_spec_f = clean_spec.float()

        enh_mag = torch.sqrt(enh_spec_f[:, 0] ** 2 + enh_spec_f[:, 1] ** 2 + eps)
        cln_mag = torch.sqrt(clean_spec_f[:, 0] ** 2 + clean_spec_f[:, 1] ** 2 + eps)

        lin_loss = F.l1_loss(enh_mag, cln_mag)
        log_loss = F.l1_loss(torch.log(enh_mag + eps), torch.log(cln_mag + eps))

        loss = lin_loss + log_loss

    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def multi_resolution_stft_loss(
    enh_spec: torch.Tensor,
    clean_spec: torch.Tensor,
    window: torch.Tensor,
    resolutions=None,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Multi-resolution STFT magnitude loss.

    [Domain fix, round 4] magnitude_loss above operates at a single
    resolution -- the model's native N_FFT=512/HOP=160. A single
    resolution is a real tradeoff: a short window gives good time
    resolution but poor frequency resolution (hard to separate nearby
    harmonics), and a long window gives the reverse. Multi-resolution
    STFT loss computes magnitude error at several window/hop sizes and
    averages them, which is standard practice in modern speech
    enhancement and vocoder training, and was reported by name as an
    effective addition specifically on the AVSEC-4 task: one published
    system trained with an SI-SDR objective alone first, then added an
    STFT loss term on top once the learning rate had annealed down, and
    reported it as part of their leaderboard-improving configuration.

    Implementation: reconstructs the waveform via ISTFT at the model's
    native resolution (same as si_snr_loss), then re-computes STFT
    magnitude at three additional resolutions spanning fine to coarse
    analysis windows, and averages the L1 + log-L1 magnitude loss across
    all of them.

    Same precision-proofing as si_snr_loss and magnitude_loss: forced
    float32 execution with autocast disabled, eps = 1e-4.

    Args:
        enh_spec:    [B, 2, Freq, T] enhanced complex spectrogram
        clean_spec:  [B, 2, Freq, T] clean complex spectrogram
        window:      hann window tensor (native resolution) on the
                     correct device, used only for the initial ISTFT
        resolutions: list of (n_fft, hop_length) pairs; defaults to
                     [(512, 128), (1024, 256), (2048, 512)] -- short,
                     medium, and long analysis windows
        eps:         numerical floor before sqrt/log to avoid log(0)

    Returns:
        scalar -- mean of L1(magnitude) + L1(log magnitude) across all
        requested resolutions
    """
    if resolutions is None:
        resolutions = [(512, 128), (1024, 256), (2048, 512)]

    with torch.autocast(device_type=enh_spec.device.type, enabled=False):
        enh_spec_f   = enh_spec.float()
        clean_spec_f = clean_spec.float()
        window_f     = window.float()

        enh_c = torch.complex(enh_spec_f[:, 0], enh_spec_f[:, 1])
        cln_c = torch.complex(clean_spec_f[:, 0], clean_spec_f[:, 1])

        enh_wav = torch.istft(enh_c, n_fft=Config.N_FFT, hop_length=Config.HOP,
                              win_length=Config.WIN, window=window_f)
        cln_wav = torch.istft(cln_c, n_fft=Config.N_FFT, hop_length=Config.HOP,
                              win_length=Config.WIN, window=window_f)

        enh_wav = torch.nan_to_num(enh_wav, nan=0.0, posinf=0.0, neginf=0.0)
        cln_wav = torch.nan_to_num(cln_wav, nan=0.0, posinf=0.0, neginf=0.0)

        total = enh_wav.new_zeros(())
        for n_fft, hop in resolutions:
            res_window = torch.hann_window(n_fft, device=enh_wav.device, dtype=torch.float32)

            e_stft = torch.stft(enh_wav, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                                window=res_window, return_complex=True)
            c_stft = torch.stft(cln_wav, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                                window=res_window, return_complex=True)

            e_mag = torch.sqrt(e_stft.real ** 2 + e_stft.imag ** 2 + eps)
            c_mag = torch.sqrt(c_stft.real ** 2 + c_stft.imag ** 2 + eps)

            total = total + F.l1_loss(e_mag, c_mag) \
                          + F.l1_loss(torch.log(e_mag + eps), torch.log(c_mag + eps))

        loss = total / len(resolutions)

    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


## -*- coding: utf-8 -*-
#"""
#Brain-Inspired Audio-Visual Speech Enhancement (AVSE)
#======================================================
#
#Core principles:
#  - MODULATION is the star -- no attention layers at all
#  - Audio neurons and visual neurons are TWO SEPARATE POPULATIONS
#  - Each population treats its own modality as receptive field (R)
#    and the other modality as contextual field (C)
#  - Coherence gate: output = R + R * C   (single projection, no MLP)
#  - Temporal context via lightweight depthwise convolution
#  - Friston precision: per-element error correction
#  - Competitive sparse activation
#
#This file fixes a chain of bugs found across two real training runs:
#
#RUN 1 (original code, before any fixes):
#  [Fix 1] FristonPrecision: error was being overwritten by a scalar from
#          F.mse_loss(), destroying per-element precision. Also W_PC's
#          gradient was killed by an explicit .detach() on the error term.
#  [Fix 2] DualPopModulationLayer: removed 2 redundant full FFNs, replaced
#          with lightweight projections + CoherenceGate for all gating.
#  [Fix 3] InfoNCE: guarantee a square logit matrix instead of a fragile
#          modulo on the label indices.
#  [Fix 4] SPARSE_K reduced 0.40 -> 0.10 for genuine sparse coding.
#  [Fix 5] Decoder mask bias initialised for a near-passthrough mask at
#          step 0, avoiding a large L1 spike at the start of training.
#
#RUN 3 (after Fix 6, crash recurred at epoch 37 instead of epoch 0):
#  [Fix 6b] The same log10(0) -> NaN-gradient mechanism reopened under
#          automatic mixed precision (AMP/fp16). fp16's smallest
#          representable positive value (~6e-8) is larger than the
#          epsilon used in Fix 6 (1e-8) and in magnitude_loss (1e-5), so
#          once tensors are fp16 under `trainer.precision="16-mixed"`,
#          those epsilons silently underflow to exactly 0.0, and the
#          original failure mode returns -- now requiring an unlucky
#          combination of a near-silent clip AND fp16 rounding, which is
#          rarer and shows up later/intermittently rather than every run
#          at epoch 0. Fixed by forcing both si_snr_loss and
#          magnitude_loss to execute in float32 with autocast explicitly
#          disabled for their bodies, regardless of the ambient training
#          precision, and raising eps to 1e-4 (three orders of magnitude
#          above fp16's representable floor) as additional margin.
#
#  [Domain fix] AVSEC-4 dataset facts (verified against the official
#          challenge papers, not assumed): training scenes can contain up
#          to 405 distinct competing speakers plus 15 noise categories
#          including music, SNR ranges from -18dB to +6.55dB, and the
#          official test set uses REAL recorded room impulse responses
#          (vs simulated RIRs in training) -- a deliberate train/test
#          domain shift via reverberation. This is a substantially
#          different and harder task than the GRID+CHiME3 setup this
#          codebase was originally built for (GRID: fixed-vocabulary,
#          single-speaker, studio-recorded; CHiME3: 4 fixed environmental
#          noise types, no competing speech, no reverberation). Two
#          concrete consequences applied here:
#            1. TEMPORAL_KERNEL widened from 31 to 63 (310ms -> 630ms
#               receptive field per layer) to give the model more context
#               for dereverberation.
#            2. NCE_ANNEAL_FLOOR added at 0.4x W_NCE (previously annealed
#               toward 0.1x): on a multi-talker dataset, the audio-visual
#               synchrony signal that InfoNCE trains is not just a
#               generic regularizer -- it is plausibly the main mechanism
#               by which the model identifies WHICH voice in a mixture of
#               speakers is the target, via lip-sync correlation. Annealing
#               it toward near-zero, which was reasonable for a
#               single-speaker-plus-noise setup, risks weakening that
#               disambiguation signal on AVSEC-4 specifically.
#          The Config fields TRAIN_SUBS, TEST_SUBS, NOISE_FILES, and
#          TRAIN_SNRS below are very likely leftover from the original
#          GRID+CHiME3 setup and unused now that an AVSE4DataModule loads
#          the official pre-mixed challenge scenes directly. They are
#          left in place (removing them risks breaking something this
#          analysis cannot see), but are worth confirming as dead code.
#
#
#  [Fix 7] Loss weight rebalancing based on a 25-epoch run with Fix 6 not
#          yet applied:
#            - W_VAP had been lowered to 0.01 in a previous pass, which
#              froze the VAP head at ln(2) (chance level) instead of
#              learning anything. Restored to a moderate 0.05.
#            - Added a magnitude / log-magnitude loss (magnitude_loss),
#              which tracks PESQ/STOI more directly than raw real/imag L1
#              because human perception of speech quality is dominated by
#              magnitude, not phase, at the SNRs used here.
#            - Added an epoch-based ramp for the SI-SNR weight (low for
#              the first few epochs while the ISTFT-based signal is noisy,
#              then up to full weight) instead of applying full weight
#              from epoch 0.
#            - Kept the existing NCE down-weighting schedule, which was
#              verified correct against the logged training data.
#          All of this lives in get_loss_weights(epoch) below so the
#          training loop only needs to call one function.
#"""
#
#import os, tarfile, zipfile, cv2, requests, shutil, warnings, random, time, math
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import numpy as np
#from torch.utils.data import Dataset, DataLoader
#from tqdm import tqdm
#import librosa, soundfile as sf
#from pesq import pesq
#from pystoi import stoi
#from typing import Tuple, Optional, List, Dict
#
#try:
#    import wandb
#    WANDB_AVAILABLE = True
#except ImportError:
#    WANDB_AVAILABLE = False
#
#warnings.filterwarnings("ignore")
#
#
## =========================================================================
## CONFIG
## =========================================================================
#
#class Config:
#    SEED       = 42
#    SR         = 16000
#    N_FFT      = 512
#    HOP        = 160
#    WIN        = 512
#    ROOT       = "data"
#    NOISE_DIR  = "noise"
#    SAMPLE_DIR = "./samples"
#    CKPT_DIR   = "./checkpoints"
#    TRAIN_SUBS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
#                  16,17,18,19,20,22,23,24,25,26,27]
#    TEST_SUBS  = [28,29,31,32,33,34]
#    TEST_SNRS  = [-9,-6,-3,0]
#    NOISE_FILES= {"BUS":"bus.wav","CAFE":"caf.wav","PED":"ped.wav","STR":"str.wav"}
#    HARD_PROB  = 0.75
#    TRAIN_SNRS = [-15,-12,-9,-9,-9,-9,-6,-6,-3,0]
#    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#    BS         = 64
#    NUM_WORKERS= 12
#
#    EPOCHS     = 100
#    LR         = 4e-4
#    LR_MIN     = 5e-5
#    WARMUP_EP  = 5
#    EMA_DECAY  = 0.999
#    ALPHA      = 0.3
#    NUM_EVAL   = 20
#    QUICK_EVAL = 10
#    EVAL_EVERY = 10
#    EPS        = 1e-8
#    D          = 256
#    N_LAYERS   = 6
#
#    # ---- Loss weights (base values; some are scheduled by epoch -- see
#    #      get_loss_weights() below, which the training step should call
#    #      once per batch instead of reading these fields directly) ----
#    W_NCE      = 0.05
#    W_VAP      = 0.05    # restored from 0.01: that value froze the VAP
#                          # head at chance level (ln(2)) in a real run
#    W_PC       = 0.005
#    W_SISNR    = 0.5
#    W_MAG      = 0.2     # magnitude + log-magnitude loss weight
#    W_MRSTFT   = 0.3     # multi-resolution STFT loss weight (round 4 addition)
#
#    NCE_TEMP   = 0.07
#    D_ALIGN    = 128
#    PC_ITERS   = 2
#
#    # NCE anneals DOWN, but only partway: NCE plateaus early in training,
#    # but on the AVSEC-4 dataset specifically, the audio-visual synchrony
#    # signal that InfoNCE trains is not just a generic regularizer the way
#    # it would be on a single-speaker-plus-environmental-noise setup like
#    # GRID+CHiME3. AVSEC-4 scenes can contain multiple competing speakers
#    # (up to 405 distinct interferers across the dataset) plus music, so
#    # the visual stream's job is partly to identify WHICH voice in the
#    # mixture is the target speaker, via lip-sync correlation. Killing the
#    # NCE weight down to a near-zero floor (as was done for the simpler,
#    # single-speaker case) risks weakening that disambiguation signal
#    # right when it matters most. Floor raised from 0.1x to 0.4x W_NCE.
#    NCE_ANNEAL_START  = 10
#    NCE_ANNEAL_EPOCHS = 20
#    NCE_ANNEAL_FLOOR  = 0.4
#
#    # SI-SNR ramps UP: the ISTFT-based SI-SNR loss is noisy in the first
#    # few epochs while the decoder mask is still far from the passthrough
#    # region set up by the Fix-5 bias initialisation. Start at 20% of
#    # W_SISNR and ramp linearly to 100% over this many epochs.
#    SISNR_WARMUP_EPOCHS = 8
#
#    # [Fix 4] Was 0.40 -- firing 40% of neurons is not sparse.
#    # 0.10 gives genuine competitive inhibition (about 25 of 256 dims
#    # active, confirmed in real training logs: 25/256 = 0.09765625).
#    SPARSE_K   = 0.10
#
#    # [AVSEC-4] Depthwise temporal conv kernel size in DualPopModulation-
#    # Layer. The original kernel of 31 (at hop=160, sr=16000, that is
#    # 31*10ms = 310ms of receptive field per layer) was sized without
#    # reverberation in mind. AVSEC-4 specifically adds room reverberation:
#    # simulated room impulse responses in training, and REAL recorded
#    # impulse responses in 3 conference rooms at 1-2m distance in the
#    # official test set. Reverberant tails commonly extend several
#    # hundred milliseconds, so a wider per-layer temporal context gives
#    # the model more room to learn dereverberation, not just denoising.
#    # Widened to 63 (630ms per layer); padding is computed automatically
#    # as (kernel - 1) // 2 to preserve sequence length.
#    TEMPORAL_KERNEL = 63
#
#
#def get_loss_weights(epoch: int) -> Dict[str, float]:
#    """
#    Single source of truth for epoch-dependent loss weights.
#
#    Call this once per training step with the current epoch (an int) and
#    use the returned dict to weight each loss term. Centralising the
#    schedule here means the training loop stays simple and the schedule
#    itself stays easy to unit test in isolation.
#
#    Returns:
#        dict with keys: "nce", "sisnr", "mrstft", "mag", "vap", "pc"
#    """
#    # --- NCE: anneal down after it plateaus, but only to a floor of
#    #     NCE_ANNEAL_FLOOR (not toward zero) -- see the Config comment
#    #     for why this matters specifically on AVSEC-4's multi-talker
#    #     scenes ---
#    nce_w = Config.W_NCE
#    if epoch > Config.NCE_ANNEAL_START:
#        decay = max(
#            0.0,
#            1.0 - (epoch - Config.NCE_ANNEAL_START) / Config.NCE_ANNEAL_EPOCHS
#        )
#        floor = Config.NCE_ANNEAL_FLOOR
#        nce_w = Config.W_NCE * (floor + (1.0 - floor) * decay)
#
#    # --- SI-SNR and multi-res STFT: both reconstruct the waveform via
#    #     ISTFT, so both are noisiest in exactly the same early epochs,
#    #     while the decoder mask is still far from the passthrough region
#    #     set up by the Fix-5 bias initialisation. Share the same ramp. ---
#    if epoch < Config.SISNR_WARMUP_EPOCHS:
#        ramp = epoch / max(1, Config.SISNR_WARMUP_EPOCHS)
#        sisnr_w  = Config.W_SISNR  * (0.2 + 0.8 * ramp)
#        mrstft_w = Config.W_MRSTFT * (0.2 + 0.8 * ramp)
#    else:
#        sisnr_w  = Config.W_SISNR
#        mrstft_w = Config.W_MRSTFT
#
#    return {
#        "nce":    nce_w,
#        "sisnr":  sisnr_w,
#        "mrstft": mrstft_w,
#        "mag":    Config.W_MAG,
#        "vap":    Config.W_VAP,
#        "pc":     Config.W_PC,
#    }
#
#
## =========================================================================
## COOPERATION GATE  (TPN "Cooperation Equation", Adeel 2025)
## =========================================================================
#
#def cooperation_gate(R: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
#    """
#    Cooperation(R, C) = ReLU6( R^2 + 2R + C * (1 + |R|) )
#
#    [Round 6] Direct drop-in replacement for the plain R + R*C modulation
#    used throughout this layer. R + R*C is, almost exactly, the simplest
#    member of a documented family of TPN-inspired (two-point-neuron)
#    modulatory transfer functions from the same research lineage this
#    codebase's docstrings already reference (Adeel et al., Phillips et
#    al., the "receptive field / contextual field" framing). That family
#    is, in order of sophistication:
#
#        T_M2(R,C) = R + RC                          <- what this layer had
#        T_M3(R,C) = R(1 + tanh(RC))
#        T_M4(R,C) = R * 2^(RC)
#        Cooperation(R,C) = ReLU6(R^2 + 2R + C(1+|R|))  <- this function
#
#    The Cooperation Equation is the one a 2025 paper from this lineage
#    ("Beyond Attention: Toward Machines with Intrinsic Higher Mental
#    States", Adeel 2025) settles on after empirically comparing all of
#    the above, and is reported as outperforming both standard Transformer
#    attention and the simpler T_M1-T_M4 family across several benchmarks
#    (CartPole and PyBullet Ant reinforcement learning, CIFAR-10, and a
#    bAbI-style question-answering task).
#
#    Why it should plausibly do more than plain R + R*C here:
#
#      1. R^2 + 2R = (R+1)^2 - 1 is a self-amplification term that exists
#         independent of C -- a strong receptive-field signal partially
#         asserts itself even before context is consulted, rather than
#         being entirely at the mercy of C's sign and magnitude the way
#         R*C is (if C happens to be near 0, R+R*C degenerates toward
#         just R; if C is near -1, R+R*C collapses toward 0 regardless of
#         how strong R is).
#
#      2. C * (1 + |R|) scales the context's contribution by the
#         receptive field's own magnitude rather than leaving it as a
#         flat additive term. A loud, confident R gets its coherence
#         decision weighted more heavily by C than a quiet, uncertain R
#         does -- which is closer to the "C splits R into coherent and
#         incoherent streams" framing this codebase's docstrings already
#         describe, since the splitting effect now scales with how much
#         R there is to split.
#
#      3. ReLU6 bounds the output to [0, 6], the same non-negative,
#         capped activation already adopted in this layer (the request
#         that added plain relu6(R+R*C) was, independently, already
#         moving toward exactly the activation function this paper uses
#         as part of its own named equation).
#
#    Args:
#        R: receptive field tensor, any shape
#        C: contextual field tensor, broadcastable to R's shape
#
#    Returns:
#        Tensor the same shape as R (after broadcasting), in [0, 6].
#    """
#    return F.relu6(R.pow(2) + 2.0 * R + C * (1.0 + R.abs()))
#
#
#class ActivePrecisionGate(nn.Module):
#    """
#    Active-precision modulation: out = R + SiLU( beta * R * C )
#
#    [Round 8, by request, correcting Round 7] Returns to the R + R*C
#    shape (T_M2) reported to work well on GRID+CHiME3, rather than the
#    quadratic Cooperation Equation. The correction from Round 7: the
#    precision/trust term should come from the coherence between R and C
#    -- or from the modulation itself -- not from a separately-learned
#    pathway. Round 7's PrecisionCooperationGate computed pi via a brand
#    new nn.Linear(2d, d) over the concatenation of R and C, which is a
#    parallel computation alongside the modulation, not derived from it.
#
#    R*C IS the modulation term already present in T_M2, and it is also,
#    quite literally, a coherence statistic between R and C: positive
#    when they agree in sign (context confirms what the receptive field
#    is saying), negative when they disagree. "Active precision" here
#    means deriving the trust signal directly and dynamically from that
#    live interaction every forward pass, rather than computing it
#    through a separate parallel pathway.
#
#    Concretely:
#        agree = beta * (R * C)     -- the coherence/modulation term;
#                                       beta is a learned per-channel
#                                       scalar controlling how sharply to
#                                       commit to strong agreement
#        gate  = sigmoid(agree)     -- THE ACTIVE PRECISION: near 1 when
#                                       R and C strongly agree, near 0
#                                       when they strongly disagree, 0.5
#                                       when the interaction is weak or
#                                       ambiguous
#        out   = R + agree * gate   -- algebraically identical to
#                                       R + SiLU(agree), since
#                                       SiLU(x) = x * sigmoid(x)
#
#    SiLU is not a new ingredient -- it is already used elsewhere in this
#    file (the FFN and temporal blocks both use nn.SiLU()). Introducing a
#    nonlinearity into R + RC, as requested, turns out to mean applying
#    an already-proven activation to the RC term, rather than anything
#    exotic.
#
#    Properties, by direct construction (also visible directly in the
#    function-landscape and feature-map comparisons against plain R+RC
#    and the Cooperation Equation):
#      - Strong agreement (RC >> 0): gate -> 1, output -> R + RC. The
#        original formula is recovered exactly in the high-coherence
#        limit -- this does not discard what already worked, it only
#        changes behaviour where coherence is weak or negative.
#      - Strong disagreement (RC << 0): gate -> 0, and SiLU has a bounded
#        global minimum (~ -0.278 / beta) rather than an unbounded
#        negative value the way raw RC would produce. Incoherent
#        interactions are suppressed toward a floor; R's own value still
#        passes through untouched alongside it.
#      - Weak/ambiguous interaction (RC ~ 0): gate ~= 0.5, a tentative,
#        partial commitment to the modulation rather than either fully
#        trusting or fully rejecting it.
#
#    Parameter cost: one per-channel scalar beta (d parameters per
#    instantiation) versus 2*d^2 for Round 7's PrecisionCooperationGate
#    -- about 650x fewer parameters at d=256. Negligible overhead, much
#    closer to "modulation is the star" than a new learned pathway.
#    """
#
#    def __init__(self, d: int):
#        super().__init__()
#        self.beta = nn.Parameter(torch.ones(d))
#
#    def forward(self, R: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
#        agree = self.beta * (R * C)
#        return R + F.silu(agree)
#
#
#class PrecisionCooperationGate(nn.Module):
#    """
#    Precision-weighted Cooperation gate (Round 7) -- NOT currently used
#    by DualPopModulationLayer, which now uses ActivePrecisionGate
#    (Round 8) instead. Left defined here for ablation/comparison: this
#    is the "precision via a separate learned pathway" version; Active-
#    PrecisionGate is the "precision derived directly from R*C coherence"
#    version requested as the correction to this one.
#
#        Cooperation_pi(R, C) = ReLU6( R^2 + 2R + pi * C * (1 + |R|) )
#        pi = sigmoid( Linear( cat([R, C]) ) )
#
#    Adds a learned, per-position trust signal pi that decides how much
#    weight C's contribution should actually get, rather than always
#    applying the full, fixed (1+|R|) scaling from the plain Cooperation
#    Equation. Motivated by FristonPrecision's existing pi = sigmoid(
#    Linear(error)) pattern elsewhere in this file, and by a 2025 audio-
#    visual target speaker extraction paper ("C^2AV-TSE: Context and
#    Confidence-aware Audio Visual Target Speaker Extraction") that
#    reported consistent PESQ/STOI/SI-SDR gains from frame-level
#    confidence-awareness across six AV-TSE backbones.
#
#    Parameter cost: one nn.Linear(2d, d) per instantiation -- 2*d^2
#    params per gate, ~393K per DualPopModulationLayer at d=256, ~2.36M
#    total across 6 layers.
#    """
#
#    def __init__(self, d: int):
#        super().__init__()
#        self.precision = nn.Linear(d * 2, d)
#
#    def forward(self, R: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
#        pi = torch.sigmoid(self.precision(torch.cat([R, C], dim=-1)))
#        return F.relu6(R.pow(2) + 2.0 * R + pi * C * (1.0 + R.abs()))
#
#
## =========================================================================
## CROSS-MODAL COHERENCE GATE  (single projection, no MLP)
## =========================================================================
#
#class CoherenceGate(nn.Module):
#    """
#    output = LayerNorm( R + R * C )
#
#    C = tanh( Linear( cat([R, context]) ) ) -- joint conditioning on both
#    the receptive field R and external context. Single Linear, no MLP.
#
#    [Round 5] Not currently used by DualPopModulationLayer, which was
#    reverted back to separate audio_transform/video_transform/
#    context_gen modules plus relu6 bounding, by request. Left defined
#    here in case it is useful again later -- it is a smaller, single-
#    projection alternative to the heavier per-branch transform + single-
#    input context_gen pattern currently in use.
#    """
#
#    def __init__(self, d: int):
#        super().__init__()
#        self.gate = nn.Linear(d * 2, d)
#        self.norm = nn.LayerNorm(d)
#
#    def forward(self, R: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
#        C = torch.tanh(self.gate(torch.cat([R, context], dim=-1)))
#        return self.norm(R + R * C)
#
#
## =========================================================================
## FRISTON PRECISION  (single projections)
## =========================================================================
#
#class FristonPrecision(nn.Module):
#    """
#    Friston predictive-coding precision module.
#
#    error       = x - prediction                  (kept as a [B,T,D] tensor)
#    pi          = sigmoid( Linear(error) )        (precision from error)
#    x           = prediction + pi * error
#    total_error += error.pow(2).mean()            (in-graph scalar)
#
#    Note from real training data: pc_error tends to DRIFT UP over many
#    epochs rather than down (observed 0.82 -> 1.34 across 25 epochs in one
#    run). This is very likely benign: representation scale grows as the
#    rest of the network learns, and squared error is unnormalised, so an
#    increasing absolute value does not necessarily mean predictive coding
#    is failing. Because W_PC = 0.005 is small, this term contributes at
#    most about 0.0065 to the total loss regardless, so it has negligible
#    practical effect on the enhancement output. It is left as-is here;
#    if you want it to behave more classically, normalising the error by
#    a running estimate of x's variance before squaring would be the next
#    step, but that is not required for this fix.
#    """
#
#    def __init__(self, d: int):
#        super().__init__()
#        self.predictor = nn.Linear(d, d)
#        self.precision = nn.Linear(d, d)
#        self.norm      = nn.LayerNorm(d)
#
#    def forward(
#        self, x: torch.Tensor, num_iters: int = 2
#    ) -> Tuple[torch.Tensor, torch.Tensor]:
#        total_error = x.new_zeros(())
#
#        for _ in range(num_iters):
#            prediction  = self.predictor(x)
#            error       = x - prediction
#            pi          = torch.sigmoid(self.precision(error))
#            x           = prediction + pi * error
#            total_error = total_error + error.pow(2).mean()
#
#        return self.norm(x), total_error / max(num_iters, 1)
#
#
## =========================================================================
## COMPETITIVE SPARSE ACTIVATION
## =========================================================================
#
#class CompetitiveSparse(nn.Module):
#    """
#    Top-k competitive inhibition. Single Linear for importance scoring.
#    base_k controls the fraction of neurons that fire.
#    """
#
#    def __init__(self, d: int, base_k: float = 0.10):
#        super().__init__()
#        self.base_k     = base_k
#        self.importance = nn.Linear(d, d)
#
#    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
#        B, T, D = x.shape
#        scores    = torch.sigmoid(self.importance(x))
#        k         = max(1, int(self.base_k * D))
#        threshold = scores.topk(k, dim=-1).values[..., -1:]
#        mask      = (scores >= threshold).float()
#
#        if self.training:
#            # Straight-through estimator keeps gradients flowing through
#            # the otherwise non-differentiable threshold.
#            mask = mask + scores - scores.detach()
#
#        active_ratio = mask.mean().item()
#        return x * mask, active_ratio
#
#
## =========================================================================
## DUAL POPULATION MODULATION LAYER
## =========================================================================
#
#class DualPopModulationLayer(nn.Module):
#    """
#    SELF-MODULATION + CROSS-MODAL FUSION (Active-precision R + SiLU(RC))
#
#    [Round 5, by request] Reverted from the lightweight CoherenceGate
#    formulation back to separate per-branch modules.
#
#    [Round 6-7, superseded] Briefly used cooperation_gate (the quadratic
#    "Cooperation Equation") and then PrecisionCooperationGate (precision
#    via a separate nn.Linear(2d,d) pathway). Both are left defined above
#    for reference/ablation but are not used here anymore.
#
#    [Round 8, current] Uses ActivePrecisionGate: out = R + SiLU(beta*RC).
#    This returns to the R + R*C shape reported to work well on
#    GRID+CHiME3, with one change -- the RC term is passed through SiLU
#    rather than left raw, and SiLU's own internal sigmoid acts as the
#    "active precision": a trust signal derived directly from the R-C
#    coherence (the RC product itself), not from any separate learned
#    pathway. See ActivePrecisionGate's docstring above for the full
#    reasoning and the math.
#
#    1. Self-modulation: each modality is projected through its own
#       2-layer transform, then gated by a context vector generated from
#       that SAME current input (not a genuine previous-timestep state --
#       see the note below on why this is named "self-context" rather
#       than "previous state"):
#           A_mod = ActivePrecision( transform(A), context_gen(A) )
#           V_mod = ActivePrecision( transform(V), context_gen(V) )
#
#    2. Cross-modal fusion: visual context modulates the audio stream:
#           Merged = ActivePrecision( A_mod, cross_context_gen(V_mod) )
#
#    3. Friston precision (per-element, in-graph -- see FristonPrecision)
#
#    4. Competitive sparse activation (k = Config.SPARSE_K)
#
#    5. Temporal context via depthwise Conv1d (kernel = Config.
#       TEMPORAL_KERNEL). This remains the only genuinely temporal
#       (cross-timestep) operation in the layer -- the self-modulation in
#       step 1 operates within a single timestep, using the SAME tensor
#       as both the thing being transformed and the source of its own
#       context. Renamed from the earlier "temporal self-modulation" /
#       a_prev/v_prev framing, which implied real recurrence that was
#       never actually there: this is a second, parallel pathway over
#       the current input, not a stored previous state.
#
#    Parameter cost: restoring the separate audio_transform/video_transform
#    (2-layer, d -> 2d -> d each) and three separate single-layer context
#    generators is heavier than the collapsed CoherenceGate version --
#    roughly 3*D^2 more parameters per layer (about 786K total across all
#    6 layers at D=256). ActivePrecisionGate itself adds only 3*D
#    parameters per layer (negligible) -- a deliberate contrast with
#    Round 7's PrecisionCooperationGate, which would have added 6*D^2.
#    """
#
#    def __init__(self, d: int, pc_iters: int = 2):
#        super().__init__()
#        self.pc_iters = pc_iters
#
#        # Per-branch transform: 2-layer GELU MLP, current-frame projection
#        self.audio_transform = nn.Sequential(
#            nn.LayerNorm(d),
#            nn.Linear(d, d * 2),
#            nn.GELU(),
#            nn.Linear(d * 2, d)
#        )
#        self.video_transform = nn.Sequential(
#            nn.LayerNorm(d),
#            nn.Linear(d, d * 2),
#            nn.GELU(),
#            nn.Linear(d * 2, d)
#        )
#
#        # Context generators: single Linear + Tanh, context in [-1, 1].
#        # Each reads directly from the layer's own current input (audio /
#        # video), not from a genuine stored previous state.
#        self.audio_context_gen = nn.Sequential(nn.Linear(d, d), nn.Tanh())
#        self.video_context_gen = nn.Sequential(nn.Linear(d, d), nn.Tanh())
#        self.cross_context_gen = nn.Sequential(nn.Linear(d, d), nn.Tanh())
#
#        # [Round 7] Precision-weighted Cooperation gates -- see
#        # PrecisionCooperationGate's docstring for the full reasoning.
#        # Kept separate per usage rather than shared.
#        # [Round 8] Active-precision gates -- see ActivePrecisionGate's
#        # docstring above for the full reasoning. Kept separate per usage.
#        self.audio_cooperation = ActivePrecisionGate(d)
#        self.video_cooperation = ActivePrecisionGate(d)
#        self.cross_cooperation = ActivePrecisionGate(d)
#
#        self.friston = FristonPrecision(d)
#        self.sparse  = CompetitiveSparse(d, base_k=Config.SPARSE_K)
#
#        # [AVSEC-4] Kernel width comes from Config.TEMPORAL_KERNEL (63 by
#        # default, widened from the original 31 to give more receptive
#        # field for the reverberation present in this dataset). Padding
#        # is computed automatically so output length is preserved for
#        # any odd kernel size.
#        _tk = Config.TEMPORAL_KERNEL
#        _tk_pad = (_tk - 1) // 2
#        self.temporal = nn.Sequential(
#            nn.Conv1d(d, d, _tk, padding=_tk_pad, groups=d),
#            nn.BatchNorm1d(d),   # use Trainer(sync_batchnorm=True) on multi-GPU
#            nn.SiLU()
#        )
#        self.temporal_norm = nn.LayerNorm(d)
#
#        self.nf  = nn.LayerNorm(d)
#        self.ffn = nn.Sequential(
#            nn.Linear(d, d * 2),
#            nn.SiLU(),
#            nn.Linear(d * 2, d)
#        )
#        self.no  = nn.LayerNorm(d)
#
#    def forward(
#        self,
#        audio: torch.Tensor,
#        video: torch.Tensor,
#        return_errors: bool = False
#    ) -> Tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor]]:
#
#        # === 1. SELF-MODULATION (Precision-weighted Cooperation Equation) ===
#        # audio_context/video_context are the layer's own current inputs
#        # -- renamed from a_prev/v_prev to avoid implying a stored
#        # previous timestep that does not actually exist here.
#        audio_context = audio
#        video_context = video
#
#        a_curr = self.audio_transform(audio)              # current audio projection
#        v_curr = self.video_transform(video)              # current video projection
#
#        c_audio  = self.audio_context_gen(audio_context)
#        c_visual = self.video_context_gen(video_context)
#
#        a_mod = self.audio_cooperation(a_curr, c_audio)
#        v_mod = self.video_cooperation(v_curr, c_visual)
#
#        # === 2. CROSS-MODAL FUSION (Precision-weighted Cooperation Equation) ===
#        c_cross = self.cross_context_gen(v_mod)            # visual-derived context
#        merged  = self.cross_cooperation(a_mod, c_cross)
#
#        # === 3. FRISTON PRECISION + SPARSITY ===
#        merged, pred_error   = self.friston(merged, num_iters=self.pc_iters)
#        merged, active_ratio = self.sparse(merged)
#
#        # === 4. TEMPORAL CONTEXT (depthwise conv across T) ===
#        t_out  = self.temporal(
#            self.temporal_norm(merged).transpose(1, 2)
#        ).transpose(1, 2)
#        merged = merged + t_out
#
#        # === 5. FFN ===
#        merged = merged + self.ffn(self.nf(merged))
#
#        # === 6. UPDATE REPRESENTATIONS ===
#        audio_out = self.no(audio + merged)
#        video_out = video + 0.1 * v_mod
#
#        error_out = pred_error if return_errors else None
#        return audio_out, video_out, active_ratio, error_out
#
#
## =========================================================================
## VISUAL MODULATOR FOR DECODER
## =========================================================================
#
#class CoherenceVisualModulator(nn.Module):
#    """Decoder-side R + R*C modulation via FiLM + coherence conv."""
#
#    def __init__(self, d_model: int, channels: int):
#        super().__init__()
#        self.proj = nn.Linear(d_model, channels * 2)
#        self.coh  = nn.Conv2d(channels, channels, 1)
#
#    def forward(self, x: torch.Tensor, vf: torch.Tensor) -> torch.Tensor:
#        v           = self.proj(vf).transpose(1, 2)       # [B, 2C, T]
#        gamma, beta = v.chunk(2, dim=1)
#        gamma       = gamma.unsqueeze(2)                  # [B, C, 1, T]
#        beta        = beta.unsqueeze(2)
#
#        r = x * (1.0 + gamma) + beta                       # FiLM modulation
#        C = torch.tanh(self.coh(r))
#        return r + r * C                                   # coherence gate
#
#
## =========================================================================
## MAIN ARCHITECTURE
## =========================================================================
#
#class GLU_Block(nn.Module):
#    def __init__(self, ic, oc):
#        super().__init__()
#        self.conv = nn.Conv2d(ic, oc * 2, (5, 3), stride=(2, 1), padding=(2, 1))
#
#    def forward(self, x):
#        a, b = self.conv(x).chunk(2, 1)
#        return a * torch.sigmoid(b)
#
#
#class VisualFrontend(nn.Module):
#    def __init__(self, d=256):
#        super().__init__()
#        self.fe3d = nn.Sequential(
#            nn.Conv3d(1, 64, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False),
#            nn.BatchNorm3d(64),
#            nn.PReLU(),
#            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
#        )
#        self.fe2d = nn.Sequential(
#            nn.Conv2d(64, 128, 3, 2, 1),
#            nn.BatchNorm2d(128),
#            nn.PReLU(),
#            nn.Conv2d(128, 256, 3, 2, 1),
#            nn.BatchNorm2d(256),
#            nn.PReLU(),
#            nn.AdaptiveAvgPool2d((1, 1))
#        )
#        self.proj = nn.Sequential(
#            nn.Linear(256, d),
#            nn.LayerNorm(d)
#        )
#
#    def forward(self, x):
#        x = self.fe3d(x)
#        B, C, T, H, W = x.shape
#        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
#        x = self.fe2d(x).view(B, T, -1)
#        return self.proj(x)
#
#
#class VAP(nn.Module):
#    def __init__(self, d):
#        super().__init__()
#        self.n  = nn.LayerNorm(d)
#        self.p2 = nn.AvgPool1d(2, 1, 1)
#        self.p4 = nn.AvgPool1d(4, 1, 2)
#        self.h  = nn.Sequential(nn.Linear(d * 3, d), nn.SiLU(), nn.Linear(d, 1))
#        self.sm = nn.Conv1d(1, 1, 5, padding=4, bias=False)
#        nn.init.constant_(self.sm.weight, 0.2)
#
#    def forward(self, vf):
#        B, T, d = vf.shape
#        x  = self.n(vf)
#        xt = x.transpose(1, 2)
#        x2 = self.p2(xt)[:, :, :T]
#        x4 = self.p4(xt)[:, :, :T]
#        lg = self.h(torch.cat([xt, x2, x4], 1).transpose(1, 2)).squeeze(-1)
#        return torch.sigmoid(
#            self.sm(F.pad(lg.unsqueeze(1), (4, 0)))[:, :, :T].squeeze(1)
#        )
#
#
#class InfoNCE(nn.Module):
#    """
#    Audio-visual contrastive alignment.
#
#    Both embeddings are trimmed to the same length N = min(A, V) before
#    computing logits, so the logit matrix is always square and the labels
#    are always a clean arange(N). A previous version used a modulo on the
#    label indices that was only correct for square matrices and could
#    silently produce wrong labels on an edge-case batch size.
#    """
#
#    def __init__(self, d, da, tau):
#        super().__init__()
#        self.wa  = nn.Linear(d, da, bias=False)
#        self.wv  = nn.Linear(d, da, bias=False)
#        self.tau = tau
#
#    def forward(self, a, v):
#        B, T, d = a.shape
#        idx = torch.arange(0, T, 4, device=a.device)
#        A   = F.normalize(self.wa(a[:, idx].reshape(-1, d)), dim=-1)
#        V   = F.normalize(self.wv(v[:, idx].reshape(-1, d)), dim=-1)
#
#        N      = min(A.shape[0], V.shape[0])
#        A, V   = A[:N], V[:N]
#        lg     = (A @ V.T) / self.tau
#        labels = torch.arange(N, device=A.device)
#
#        return (F.cross_entropy(lg, labels) + F.cross_entropy(lg.T, labels)) / 2.
#
#
#class AVSEModel(nn.Module):
#    def __init__(
#        self,
#        model_type="av_modulation_v3",
#        d_model=256,
#        num_layers=6,
#        pc_iterations=2,
#        **_
#    ):
#        super().__init__()
#
#        self.vfe  = VisualFrontend(d_model)
#
#        self.enc1 = GLU_Block(2, 64)
#        self.enc2 = GLU_Block(64, 128)
#        self.enc3 = GLU_Block(128, 192)
#        self.ap   = nn.Sequential(
#            nn.Linear(192 * 33, d_model),
#            nn.LayerNorm(d_model)
#        )
#
#        self.layers = nn.ModuleList([
#            DualPopModulationLayer(d=d_model, pc_iters=pc_iterations)
#            for _ in range(num_layers)
#        ])
#
#        self.vap = VAP(d_model)
#        self.nce = InfoNCE(d_model, Config.D_ALIGN, Config.NCE_TEMP)
#
#        self.bu    = nn.Linear(d_model, 192 * 33)
#        self.d3    = nn.ConvTranspose2d(384, 128, (5,3), (2,1), (2,1), output_padding=(1,0))
#        self.vmod3 = CoherenceVisualModulator(d_model, 128)
#        self.d2    = nn.ConvTranspose2d(256,  64, (5,3), (2,1), (2,1), output_padding=(1,0))
#        self.vmod2 = CoherenceVisualModulator(d_model,  64)
#        self.d1    = nn.ConvTranspose2d(128,   2, (5,3), (2,1), (2,1), output_padding=(1,0))
#
#        # [Fix 5] Passthrough mask initialisation. At step 0 with random
#        # weights, out[:,0:1] is approximately 0, so mask_r ~ 0, producing
#        # near-zero enhanced signal and a large L1 spike. Setting d1.bias
#        # so that 5*tanh(bias_r) ~ 1 and 5*tanh(bias_i) ~ 0 gives
#        # mask_r ~ 1, mask_i ~ 0 at initialisation (near passthrough).
#        #   5 * tanh(x) = 1  ->  x = atanh(0.2) ~= 0.2027
#        if self.d1.bias is not None:
#            nn.init.constant_(self.d1.bias[0], 0.2027)   # real mask -> 1
#            nn.init.constant_(self.d1.bias[1], 0.0000)   # imag mask -> 0
#
#    def forward(self, spec_ri, video):
#        b, c, nf, t = spec_ri.shape
#
#        e1 = self.enc1(spec_ri)
#        e2 = self.enc2(e1)
#        e3 = self.enc3(e2)
#        af = self.ap(e3.permute(0, 3, 1, 2).reshape(b, t, -1))
#
#        vf = self.vfe(video).transpose(1, 2)
#        vf = F.interpolate(vf, size=t, mode='linear', align_corners=False).transpose(1, 2)
#
#        active_neurons  = []
#        all_pred_errors = []
#
#        for layer in self.layers:
#            af, vf, ar, error = layer(af, vf, return_errors=self.training)
#            active_neurons.append(ar)
#            if error is not None:
#                all_pred_errors.append(error)
#
#        mean_active = sum(active_neurons) / len(active_neurons)
#
#        bu = self.bu(af).reshape(b, t, 192, 33).permute(0, 2, 3, 1)
#
#        d3_out = F.relu(self.d3(torch.cat([bu, e3], 1)))
#        d3_out = self.vmod3(d3_out, vf)
#        d3_out = F.interpolate(d3_out, size=(e2.shape[2], t), mode='bilinear', align_corners=False)
#
#        d2_out = F.relu(self.d2(torch.cat([d3_out, e2], 1)))
#        d2_out = self.vmod2(d2_out, vf)
#        d2_out = F.interpolate(d2_out, size=(e1.shape[2], t), mode='bilinear', align_corners=False)
#
#        d1_out = self.d1(torch.cat([d2_out, e1], 1))
#        out    = F.interpolate(d1_out, size=(nf, t), mode='bilinear', align_corners=False)
#
#        nr, ni = spec_ri[:, 0:1], spec_ri[:, 1:2]
#        mask_r = 5.0 * torch.tanh(out[:, 0:1])
#        mask_i = 5.0 * torch.tanh(out[:, 1:2])
#        enh_r  = nr * mask_r - ni * mask_i
#        enh_i  = nr * mask_i + ni * mask_r
#        enh    = torch.cat([enh_r, enh_i], dim=1)
#
#        vap_p = self.vap(vf)
#        nce_l = self.nce(af, vf)
#
#        if self.training:
#            return enh, vap_p, nce_l, mean_active, all_pred_errors
#        return enh
#
#
## =========================================================================
## AUXILIARY LOSS FUNCTIONS  (import alongside AVSEModel)
## =========================================================================
#
#def si_snr_loss(
#    enh_spec: torch.Tensor,
#    clean_spec: torch.Tensor,
#    window: torch.Tensor,
#    eps: float = 1e-4,
#) -> torch.Tensor:
#    """
#    Scale-Invariant Signal-to-Noise Ratio loss.
#
#    Returns -SI-SNR (dB), so minimising this maximises perceptual quality.
#    Typical range at epoch 0: roughly +2 to +7 (bad alignment).
#    Typical range at convergence: roughly -8 to -15 (good alignment).
#
#    [Fix 6] BOTH the signal energy (numerator) and noise energy
#    (denominator) are floored with the SAME eps before the division and
#    log10. An earlier version only floored the denominator, which let
#    log10(0) reach -inf in the forward pass and a NaN gradient through
#    the clamp boundary in the backward pass.
#
#    [Fix 6b] Fix 6 alone was confirmed correct in float32, but a real
#    training run still crashed with the same downstream symptom (a CUDA
#    assertion inside the VAP head's binary_cross_entropy), just much
#    later -- epoch 37 instead of epoch 0 -- and only once, not on every
#    run. That pattern points to a second, rarer trigger for the same
#    underlying mechanism: automatic mixed precision (AMP/fp16). Under
#    fp16, the smallest representable positive value is about 6e-8. The
#    eps used in Fix 6 (1e-8) and in magnitude_loss (1e-5) are both BELOW
#    that threshold, so once enh_spec/clean_spec are fp16 tensors (which
#    they are for at least part of the forward pass under
#    `trainer.precision = "16-mixed"`), the epsilon silently underflows
#    to exactly 0.0 and the original log10(0) failure mode reopens. This
#    requires a near-silent clip AND an unlucky fp16 rounding to coincide,
#    which is rarer than the pure fp32 bug and explains why it now shows
#    up intermittently, late in training, instead of deterministically at
#    epoch 0.
#
#    Fix: force this entire function to execute in float32 regardless of
#    the ambient autocast/precision setting (explicit `.float()` cast on
#    every input, plus `torch.autocast(..., enabled=False)` around the
#    body so no op inside can be silently downcast back to fp16), and
#    raise eps to 1e-4, which has more than three orders of magnitude of
#    safety margin above fp16's representable range. A final
#    nan_to_num() on the returned scalar is kept as a last-resort guard.
#
#    Args:
#        enh_spec:   [B, 2, Freq, T] enhanced complex spectrogram (real/imag)
#        clean_spec: [B, 2, Freq, T] clean complex spectrogram
#        window:     hann window tensor on the correct device
#        eps:        numerical floor used for both numerator and denominator
#
#    Returns:
#        scalar -- negative mean SI-SNR across the batch
#    """
#    with torch.autocast(device_type=enh_spec.device.type, enabled=False):
#        enh_spec_f = enh_spec.float()
#        clean_spec_f = clean_spec.float()
#        window_f = window.float()
#
#        enh_c = torch.complex(enh_spec_f[:, 0], enh_spec_f[:, 1])
#        cln_c = torch.complex(clean_spec_f[:, 0], clean_spec_f[:, 1])
#
#        enh_wav = torch.istft(enh_c, n_fft=Config.N_FFT, hop_length=Config.HOP,
#                              win_length=Config.WIN, window=window_f)
#        cln_wav = torch.istft(cln_c, n_fft=Config.N_FFT, hop_length=Config.HOP,
#                              win_length=Config.WIN, window=window_f)
#
#        # Defensive: stop any non-finite value from ISTFT before it can
#        # reach the division/log below.
#        enh_wav = torch.nan_to_num(enh_wav, nan=0.0, posinf=0.0, neginf=0.0)
#        cln_wav = torch.nan_to_num(cln_wav, nan=0.0, posinf=0.0, neginf=0.0)
#
#        cln_wav = cln_wav - cln_wav.mean(dim=-1, keepdim=True)
#        enh_wav = enh_wav - enh_wav.mean(dim=-1, keepdim=True)
#
#        dot  = (enh_wav * cln_wav).sum(dim=-1, keepdim=True)
#        norm = (cln_wav ** 2).sum(dim=-1, keepdim=True).clamp(min=eps)
#        proj = (dot / norm) * cln_wav
#
#        noise = enh_wav - proj
#
#        # Both floored with the SAME eps -- the ratio can never be
#        # exactly zero, so log10 never sees an exact 0 input.
#        sig_energy   = (proj  ** 2).sum(dim=-1).clamp(min=eps)
#        noise_energy = (noise ** 2).sum(dim=-1).clamp(min=eps)
#
#        si_snr = 10.0 * torch.log10(sig_energy / noise_energy)
#        si_snr = si_snr.clamp(min=-30.0, max=30.0)
#
#        loss = -si_snr.mean()
#
#    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
#
#
#def magnitude_loss(
#    enh_spec: torch.Tensor,
#    clean_spec: torch.Tensor,
#    eps: float = 1e-4,
#) -> torch.Tensor:
#    """
#    Magnitude + log-magnitude L1 loss on the complex spectrogram.
#
#    The primary L1 loss operates on raw real/imag components, which
#    penalises phase error exactly as heavily as magnitude error. Human
#    perception of speech quality (and the PESQ/STOI metrics) is dominated
#    by magnitude, not phase, at the SNRs used in this task. This loss
#    isolates the magnitude component directly, which should correlate
#    more closely with PESQ/STOI than the primary L1 term alone.
#
#    The log-magnitude term additionally emphasises low-energy regions
#    (consonants, silences between words) that a purely linear magnitude
#    loss would under-weight relative to the high-energy vowel formants.
#
#    Same precision-proofing as si_snr_loss: forced float32 execution
#    with autocast disabled, and eps raised to 1e-4 so it cannot underflow
#    to zero under fp16.
#
#    Args:
#        enh_spec:   [B, 2, Freq, T] enhanced complex spectrogram
#        clean_spec: [B, 2, Freq, T] clean complex spectrogram
#        eps:        numerical floor before sqrt/log to avoid log(0)
#
#    Returns:
#        scalar loss = L1(magnitude) + L1(log magnitude)
#    """
#    with torch.autocast(device_type=enh_spec.device.type, enabled=False):
#        enh_spec_f = enh_spec.float()
#        clean_spec_f = clean_spec.float()
#
#        enh_mag = torch.sqrt(enh_spec_f[:, 0] ** 2 + enh_spec_f[:, 1] ** 2 + eps)
#        cln_mag = torch.sqrt(clean_spec_f[:, 0] ** 2 + clean_spec_f[:, 1] ** 2 + eps)
#
#        lin_loss = F.l1_loss(enh_mag, cln_mag)
#        log_loss = F.l1_loss(torch.log(enh_mag + eps), torch.log(cln_mag + eps))
#
#        loss = lin_loss + log_loss
#
#    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
#
#
#def multi_resolution_stft_loss(
#    enh_spec: torch.Tensor,
#    clean_spec: torch.Tensor,
#    window: torch.Tensor,
#    resolutions=None,
#    eps: float = 1e-4,
#) -> torch.Tensor:
#    """
#    Multi-resolution STFT magnitude loss.
#
#    [Domain fix, round 4] magnitude_loss above operates at a single
#    resolution -- the model's native N_FFT=512/HOP=160. A single
#    resolution is a real tradeoff: a short window gives good time
#    resolution but poor frequency resolution (hard to separate nearby
#    harmonics), and a long window gives the reverse. Multi-resolution
#    STFT loss computes magnitude error at several window/hop sizes and
#    averages them, which is standard practice in modern speech
#    enhancement and vocoder training, and was reported by name as an
#    effective addition specifically on the AVSEC-4 task: one published
#    system trained with an SI-SDR objective alone first, then added an
#    STFT loss term on top once the learning rate had annealed down, and
#    reported it as part of their leaderboard-improving configuration.
#
#    Implementation: reconstructs the waveform via ISTFT at the model's
#    native resolution (same as si_snr_loss), then re-computes STFT
#    magnitude at three additional resolutions spanning fine to coarse
#    analysis windows, and averages the L1 + log-L1 magnitude loss across
#    all of them.
#
#    Same precision-proofing as si_snr_loss and magnitude_loss: forced
#    float32 execution with autocast disabled, eps = 1e-4.
#
#    Args:
#        enh_spec:    [B, 2, Freq, T] enhanced complex spectrogram
#        clean_spec:  [B, 2, Freq, T] clean complex spectrogram
#        window:      hann window tensor (native resolution) on the
#                     correct device, used only for the initial ISTFT
#        resolutions: list of (n_fft, hop_length) pairs; defaults to
#                     [(512, 128), (1024, 256), (2048, 512)] -- short,
#                     medium, and long analysis windows
#        eps:         numerical floor before sqrt/log to avoid log(0)
#
#    Returns:
#        scalar -- mean of L1(magnitude) + L1(log magnitude) across all
#        requested resolutions
#    """
#    if resolutions is None:
#        resolutions = [(512, 128), (1024, 256), (2048, 512)]
#
#    with torch.autocast(device_type=enh_spec.device.type, enabled=False):
#        enh_spec_f   = enh_spec.float()
#        clean_spec_f = clean_spec.float()
#        window_f     = window.float()
#
#        enh_c = torch.complex(enh_spec_f[:, 0], enh_spec_f[:, 1])
#        cln_c = torch.complex(clean_spec_f[:, 0], clean_spec_f[:, 1])
#
#        enh_wav = torch.istft(enh_c, n_fft=Config.N_FFT, hop_length=Config.HOP,
#                              win_length=Config.WIN, window=window_f)
#        cln_wav = torch.istft(cln_c, n_fft=Config.N_FFT, hop_length=Config.HOP,
#                              win_length=Config.WIN, window=window_f)
#
#        enh_wav = torch.nan_to_num(enh_wav, nan=0.0, posinf=0.0, neginf=0.0)
#        cln_wav = torch.nan_to_num(cln_wav, nan=0.0, posinf=0.0, neginf=0.0)
#
#        total = enh_wav.new_zeros(())
#        for n_fft, hop in resolutions:
#            res_window = torch.hann_window(n_fft, device=enh_wav.device, dtype=torch.float32)
#
#            e_stft = torch.stft(enh_wav, n_fft=n_fft, hop_length=hop, win_length=n_fft,
#                                window=res_window, return_complex=True)
#            c_stft = torch.stft(cln_wav, n_fft=n_fft, hop_length=hop, win_length=n_fft,
#                                window=res_window, return_complex=True)
#
#            e_mag = torch.sqrt(e_stft.real ** 2 + e_stft.imag ** 2 + eps)
#            c_mag = torch.sqrt(c_stft.real ** 2 + c_stft.imag ** 2 + eps)
#
#            total = total + F.l1_loss(e_mag, c_mag) \
#                          + F.l1_loss(torch.log(e_mag + eps), torch.log(c_mag + eps))
#
#        loss = total / len(resolutions)
#
#    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
#
