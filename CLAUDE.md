# Project Context: AVSEC-4 Brain-Inspired Audio-Visual Speech Enhancement

Read this fully before making any changes. Then verify it against the actual
files in this directory (new_core.py, new_model.py, train.py, test.py,
submit_avse4_train.sh, submit_avse4_test.sh) -- this document is a summary
of decisions made over a long design conversation, the CODE is the ground
truth if the two ever disagree.

## What this project is

An audio-visual speech enhancement (AVSE) model for the 4th COG-MHEAR
AVSEC-4 Challenge. The task: given a noisy mixture (target speaker plus
competing speakers, environmental noise, and music, at SNR -18dB to
+6.55dB, with room reverberation) plus synchronized video of the target
speaker's face, reconstruct the clean target speech. Evaluated on PESQ,
STOI, and SI-SDR.

AVSEC-4 is NOT a simple denoising task. Key dataset facts that shape every
design decision below: training scenes can contain up to 405 distinct
competing speakers, 15 noise categories including music, and the official
test set uses REAL recorded room impulse responses while training only
sees simulated ones (a deliberate train/test domain shift via
reverberation). This is closer to "cocktail party" target speaker
extraction than denoising.

Calibration: the official baseline scores around PESQ 1.21-1.45 (often
BELOW the raw noisy input, which scores ~1.30). The current state-of-the-
art, AVSEMamba (1st place on the monaural leaderboard), reaches PESQ 2.97
on the dev set using a Mamba (selective state-space) temporal backbone.
This codebase has reached approximately PESQ 1.70-1.72 / STOI 0.65-0.66 on
a full dev-set evaluation as of the last training run referenced in this
document -- meaningfully above baseline, well short of SOTA, with no
single obvious remaining bug (the architecture trains stably and is still
slowly improving at the point training was last stopped).

## Non-negotiable architectural identity

This is a deliberate research choice, not a placeholder to be "fixed" by
adding standard Transformer attention:

- **No attention anywhere.** No softmax over sequence positions, no QKV
  similarity matrices, no quadratic-cost mechanism of any kind.
- **Modulation is the star.** All cross-modal fusion and temporal context
  is built from elementwise multiplicative/additive gating between a
  receptive field (R) and a contextual field (C), inspired by two-point-
  neuron (TPN) models of cortical pyramidal cells, where basal dendrites
  provide the feedforward receptive-field signal and apical dendrites
  provide top-down/lateral contextual modulation. This codebase's
  research lineage traces to Ahsan Adeel's lab (University of Stirling)
  and their TPN / "Cooperation Equation" line of work.
- **Coherence gates, not attention, decide relevance.** The stated
  philosophy throughout: coherent (R,C agree) information is amplified,
  incoherent (R,C disagree) information is suppressed, and "precision"
  (how much to trust this gating decision right now) should be derived
  from the coherence between R and C itself -- NOT from a separately
  learned black-box pathway. This specific point was corrected once
  already in this project's history (see ActivePrecisionGate below); do
  not reintroduce a separate learned precision pathway without being
  asked.
- **Complexity must stay O(N) / O(L) in sequence length**, same class as
  a fixed-kernel convolution, explicitly to stay competitive with (not
  copy) Mamba-style state-space approaches used by the current SOTA
  system on this exact dataset, while keeping the modulation-only
  identity.
- When asked to change the modulation equation or add a new mechanism,
  the expectation is multiple formulations are kept side by side in the
  code (old ones left defined but unused, clearly marked) rather than
  deleted, specifically so they can be compared/ablated later. Follow
  this pattern unless told otherwise.

## File map

- **new_core.py** -- model architecture (AVSEModel, DualPopModulationLayer,
  all modulation gate variants, loss functions, Config).
- **new_model.py** -- PyTorch Lightning wrapper (AVSE4BaselineModule):
  training/validation steps, loss weighting schedule, EMA, NaN guard,
  optimizer/scheduler.
- **train.py** -- Hydra entry point for training.
- **test.py** -- Hydra entry point for dev/test-set evaluation, produces
  evaluation_report.csv with per-file PESQ/STOI/SI-SDR.
- **submit_avse4_train.sh / submit_avse4_test.sh** -- SLURM sbatch
  scripts. Training script has auto-resume built in (see below).

## Key design decisions already made (do not silently re-litigate these)

**Modulation equation, current state**: `DualPopModulationLayer` uses
`ActivePrecisionGate`: `out = R + SiLU(beta * R * C)`. `beta` is a learned
per-channel scalar; the only learned parameter, deliberately minimal.
Precision is derived directly from `R*C` (the coherence between R and C)
via SiLU's internal sigmoid -- NOT from a separate Linear layer. Three
earlier variants are defined in the file but unused, kept for comparison:
`CoherenceGate` (single-projection), `cooperation_gate` /
`PrecisionCooperationGate` (a more complex quadratic form with a
separately-learned precision head -- this was explicitly corrected away
from per a direct request; the correction was "precision should come from
the coherence between R and C, or from the modulation, not a separate
pathway").

**Temporal context, current state**: two parallel paths, summed
(both implemented and verified passing a forward/backward sanity check).
`self.temporal` -- a depthwise Conv1d (kernel = `Config.TEMPORAL_KERNEL`,
currently 63, widened from 31 specifically because AVSEC-4 has real
reverberation needing more context per layer) for local, fixed-receptive-
field mixing. `self.recurrence` -- `ContextGatedRecurrence` (Round 9,
implemented), a linear-time (O(T)) selective recurrent scan:
`h_t = (1-pi_t)*h_{t-1} + pi_t*x_t` where
`pi_t = sigmoid(beta * x_t * C_t)`. `C_t` is `v_mod` (the modulated
visual features at the same T), so the visual stream gates which audio
frames drive state updates -- directly supporting speaker disambiguation
via lip-sync on the multi-talker AVSEC-4 scenes. `beta` is a learned
per-channel scalar (same minimal parameterisation as `ActivePrecisionGate`).
The scan runs as a sequential Python loop over T (correct O(T); no custom
CUDA kernel needed at AVSEC-4 utterance lengths). Both paths share the
same `temporal_norm(merged)` input and their outputs are summed:
`merged = merged + t_out + r_out`.

**Predictive coding**: `FristonPrecision` computes a per-element,
in-graph (NOT detached) precision-weighted prediction error,
`x = prediction + pi*error`, iterated `Config.PC_ITERS` times. An earlier
bug had this detached (zero gradient to W_PC) and using a scalar instead
of per-element error; both are fixed. `pc_error` is known to drift
upward slowly over long training runs rather than down -- this has been
observed multiple times, is currently believed benign given
`Config.W_PC = 0.005` is small, and has not been root-caused further.

**Losses** (see `get_loss_weights(epoch)` in new_core.py for the full
schedule): L1 (spectrogram) + magnitude/log-magnitude + SI-SNR (ramped
up over the first `SISNR_WARMUP_EPOCHS`) + multi-resolution STFT (same
ramp) + InfoNCE (audio-visual alignment, annealed DOWN after
`NCE_ANNEAL_START` but only to a floor of `NCE_ANNEAL_FLOOR`, deliberately
not to near-zero, because on a multi-talker dataset the AV synchrony
signal plausibly does real speaker-disambiguation work, not just generic
regularization) + VAP (voice activity prediction, auxiliary) + predictive-
coding error. `si_snr_loss` and `magnitude_loss` and
`multi_resolution_stft_loss` all force float32 execution with autocast
disabled internally -- this was a deliberate fix for a real, hard-to-
reproduce bug: under AMP/fp16, a small epsilon used to avoid log(0) can
underflow to exactly zero, reopening a NaN-gradient path that surfaces
many steps later as an unrelated CUDA assertion in a totally different
part of the model. Do not lower these epsilons without re-checking fp16
behavior directly.

**Resilience / training infrastructure**: `on_before_optimizer_step` in
new_model.py zeroes any non-finite gradient before the optimizer applies
it (a NaN anywhere should never crash an 8+ hour multi-GPU SLURM job).
EMA shadow weights are persisted into the checkpoint
(`on_save_checkpoint`/`on_load_checkpoint`) and restored on resume --
this was a real bug: EMA was being computed correctly every step but
never reaching disk, and was being silently reset on every SLURM requeue
before the fix. `train.py` has NO EarlyStopping (removed by request after
verifying mathematically, against real logged data, that it was not
actually the cause of a training run stopping early -- a SLURM walltime
limit was). Checkpointing instead tracks `val_pesq` (max) as the primary
criterion, since PESQ keeps improving well after raw L1 plateaus.
`submit_avse4_train.sh` has `--requeue` plus automatic detection of the
most recent `last.ckpt` on every invocation, so it survives walltime
limits indefinitely via SLURM auto-requeue without manual intervention.

**test.py** imports from `new_model`/`new_core` (NOT a separate stale
`model.py` that exists elsewhere in this repo from before any of these
fixes -- a real bug once caused a checkpoint/architecture mismatch crash
this way). It optionally evaluates with EMA weights if present in the
checkpoint (`USE_EMA_WEIGHTS` flag near the top of the file).
`compute_si_sdr` includes a small bounded lag search (+/- 80 samples) to
correct for an ISTFT sample-alignment artifact that was confirmed to
cause a handful of files to show catastrophically bad SI-SDR despite
genuinely good PESQ.

## Working conventions to follow

- **ASCII only** in all code, comments, and docstrings. No em-dashes, no
  smart quotes, no arrows (use `->`), no special symbols. This has been
  explicitly required multiple times.
- Before declaring any change finished: actually run it (syntax check,
  and a quick forward/backward pass with synthetic tensors checking for
  NaN/Inf in gradients) rather than asserting it should work.
- When given a metrics CSV or evaluation_report.csv, compute real
  statistics from it (per-epoch trends, distributions, correlations)
  rather than eyeballing a few rows.
- This is an active research codebase under iteration with a co-
  researcher from the lab whose published work this architecture is
  based on (Adeel et al., two-point-neuron / TPN line of work) -- treat
  proposed formulations as genuine research decisions to implement
  carefully and test, not as suggestions to second-guess on novelty
  grounds.

## Immediate orientation task

Before doing anything else this session: read new_core.py and
new_model.py in full, and confirm the current state of
`DualPopModulationLayer` and the loss-weighting schedule actually matches
the description above. Specifically check: (1) `ActivePrecisionGate` is
the active modulation gate; (2) `self.temporal` (depthwise Conv1d, kernel
63) and `self.recurrence` (`ContextGatedRecurrence`) both exist and are
summed in step 4 of the forward; (3) the loss schedule in
`get_loss_weights` matches the Config values described here. If anything
has drifted, say so explicitly before proceeding -- this document may be
stale relative to the latest edits.
