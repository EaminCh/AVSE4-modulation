# Parallel Ablation: Frequency-Awareness Experiments

Two independent experiments, both attacking the same diagnosed root cause:
the modulation layers currently run on a 256-dim vector that the encoder
produced by collapsing the entire 192x33 time-frequency bottleneck through
a single Linear. That discards the frequency structure (formants,
harmonics -- the cues that separate overlapping speakers) BEFORE modulation
runs. Both experiments restore frequency-awareness cheaply, in different
places and at different cost.

Everything lives in new_core_ablation.py (a copy of new_core.py with the
two experiments added behind Config flags). Your existing new_core.py is
untouched, so it remains the baseline.

## How to run the two jobs in parallel

Point new_model.py at the ablation core for these runs:
    change `from new_core import ...` to `from new_core_ablation import ...`
(or copy new_core_ablation.py over new_core.py in each job's own working
copy -- whichever fits your setup). Then set the flags in the Config class.

RUN 1 -- Experiment A (Frequency Warp):
    USE_FREQUENCY_WARP        = True
    USE_FREQ_AXIS_MODULATION  = False

RUN 2 -- Experiment B (Frequency-Axis Modulation):
    USE_FREQUENCY_WARP        = False
    USE_FREQ_AXIS_MODULATION  = True

(Optional control) RUN 3 -- both False reproduces the current 1.76 baseline
with this exact code, useful as the reference line for the comparison.

Submit each as its own SLURM job (separate output dirs / version numbers)
so they train simultaneously on different GPUs.

## What each experiment is, and the cost

### Experiment A: FrequencyWarp  (257 params, lowest risk)

A learnable per-frequency power-law compression on the input spectrogram
magnitude (phase untouched), applied once before the encoder:

    mag_warped[k] = mag[k] ^ gamma[k],   gamma[k] = sigmoid(raw_gamma[k]) in (0,1)

Each frequency bin gets its own learnable exponent. gamma < 1 is
compressive; per-frequency means the network can boost low-energy high-
frequency content (consonants, fricatives -- what STOI/intelligibility
rewards) relative to loud low-frequency vowels. Cochlea-inspired.

- COST: 257 parameters, one elementwise pass. Negligible compute, negligible
  memory, no change to sequence length or model shape. Fastest of the two.
- TARGETS: primarily STOI/intelligibility (the weaker of your two metrics).
- RISK: very low. The mask is still applied to the ORIGINAL (un-warped)
  spectrogram -- the warp is only a perceptual lens for the encoder's
  analysis, it never distorts the signal being enhanced.
- HYPOTHESIS IF IT WINS: the encoder was under-weighting perceptually
  important quiet/high-frequency detail; a cheap learnable frequency
  emphasis recovers it.

### Experiment B: FreqAxisModulation  (1,728 params, higher upside)

Runs the validated tanh coherence gate ALONG the frequency axis, inside the
bottleneck, BEFORE the channel flattening, with context from a depthwise
conv over neighbouring/harmonic frequency bins:

    C_harm[t,k] = depthwise_conv_over_freq(X)[t,k]
    out[t,k]    = X[t,k] * (1 + tanh(beta * X[t,k] * C_harm[t,k]))

This is the more direct attack: it lets modulation act on frequency
structure explicitly, learning "this band is coherent with its spectral
neighbourhood -> speech; incoherent -> noise". Same R*(1+tanh(beta*R*C))
gate you already validated as best-behaved, redirected onto frequency.

- COST: 1,728 parameters (one depthwise freq-conv + per-channel beta), plus
  two reshape/permute ops per forward on the 192x33 bottleneck. Still O(T*F*D),
  linear, no attention. Slightly slower than A but not dramatically.
- TARGETS: speaker separation directly (the core AVSEC-4 difficulty) ->
  should help both PESQ and STOI.
- RISK: low-moderate. Larger change than A, but kept inside the bottleneck
  so it does not disturb the decoder or the existing modulation layers.
- HYPOTHESIS IF IT WINS: the diagnosed root cause is correct -- modulation
  needs to act on frequency structure, not just the collapsed channel
  vector. If B clearly beats A and baseline, that validates building a
  fuller dual-axis (time AND frequency) modulation as the next step.

## What to report back

For each run, the best val_pesq / val_stoi and the epoch reached, plus the
metrics CSV. The interesting comparisons:
  - Does either beat the ~1.76 baseline, and by how much?
  - A (cheap, STOI-targeted) vs B (richer, separation-targeted): which moves
    which metric?
  - For A: check whether the learned gamma values actually moved away from
    their 0.85 init (if they stayed flat, the warp was not useful).
  - For B: check whether pc_error and the curves look healthier or just
    flatter.

This is the project's first genuinely controlled ablation -- the results will tell
us which direction is worth the bigger dual-axis refactor.