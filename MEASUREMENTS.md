# mlx-sam-audio-swift — measurements

M5 Max, 128 GB; release build; mlx-swift from 0.32.3. Oracle: upstream `facebookresearch/sam-audio` @ `bb4c699` on the
gated `facebook/sam-audio-base` checkpoint, PyTorch 2.8.0 CPU fp32. Lanes: `samaudio-gates`, fp32, TF32 off, relative
L2 as dB. Four cases (rain, speech over a song, drums in a trio, a cat with four `+` spans). 2026-11-11.

## Parity (58 / 58 lanes)

| Lane | Result | Floor |
|---|---|---|
| Key contract, `model` + `text_encoder` (codec: the weight bag is consumed exactly at load) | 0 missing, 0 unused | exact |
| T5-base encoder | 116.7–118.5 dB | 100 |
| One DiT velocity (golden inputs) | 110.7–112.3 dB | 100 |
| 16-step midpoint solve | 115.6–117.6 dB | 100 |
| Codec encode | 94.7–113.7 dB | 90 |
| Decoder blocks | 116.5–122.7 dB | 90 |
| Plain waveform | 116.7–121.4 dB | 90 |
| Watermarked waveform, fixed message | 116.7–121.4 dB | 90 |
| The watermark signal alone (marked − plain) | 92.6–103.3 dB | 60 |
| Windowed decode against the golden | 116.7–121.4 dB | 90 |
| Windowed decode against the single pass | identical (0 difference) | 100 |
| End to end: tokens and anchors | equal | exact |
| End to end: wav + prompt → marked target / residual | 97.0–110.7 dB | 90 |

Half-precision DiT against the fp32 goldens (`tier` lane): velocity 51.6–54.1 dB, solve 47.7–57.7 dB.

## Separation quality through the Swift pipeline

The pre-registered gates of `WIP/sam-audio-eval/MEASUREMENTS.md` (58 separations of 10 s mixtures with known sources,
noun prompts, one draw, outputs watermarked as shipped; metric: gain in mel-spectrogram SNR over the unprocessed
mixture; "catastrophic" = worse than the mixture):

| Tier | General sound (24) | Speech against music (8) | Stems (18) | Per 10 s clip | `phys_footprint` after load / peak |
|---|---|---|---|---|---|
| fp32 | 12.47 dB, 91.7 % ≥ 3 dB, 1 catastrophic — pass | 13.94 dB, 0 — pass | 7.68 dB, 1 | 4.1–8.6 s, median 6.90 s | 5.64 / 10.02 GB |
| **fp16 DiT** (T5 and codec fp32) | 12.47 dB, 95.8 % ≥ 3 dB, 1 catastrophic — pass | 13.94 dB, 0 — pass | 7.68 dB, 1 | 4.4–5.2 s, median 4.44 s | 4.48 / 8.60 GB |

Per stem: drums 13.5 dB, bass 5.8 dB, guitar 1.7 dB (guitar does not separate). Music against speech 15.41 dB, 0 of 8
catastrophic. No empty targets in either tier. Target + residual against the mixture: SI-SDR 8.6 dB (not
sample-aligned; `regenerated`).

## Found on the way

- **A first fp16 run failed outright** (general sound −2.85 dB median, 13 of 24 catastrophic): the RMS norms squared
  half-precision activations and overflowed. Upstream takes the norm in fp32; so does the port now, and the T5 encoder
  stays fp32. Kept as `results-swift-fp16-run1-overflow.json` in the eval directory.
- **Decoding 10 s in one pass peaks near 17 GB per waveform** (38 GB for both at once). The production decode runs the
  main path in windows of 25 latent frames with 12 frames of context, which is identical to the single pass to the
  bit and brings a 10 s decode with watermark to 3.8–5.5 GB.
- The fp32 run's per-clip time drifted from 4.1 s to 8.6 s and settled near 6.9 s with flat memory; the fp16 run stayed
  at 4.4–5.2 s. Not explained. The 16-step solve alone is 1.6–2.0 s at fp32; the rest is the two decodes and the
  watermark's four LSTM passes of 1,500 steps each, which are a plain Swift loop and the first thing to optimise.

## Not measured

int8 and smaller tiers; clips longer or shorter than 10 s; long-form windowing of the whole separation; the small and
large checkpoints; a head-to-head with Demucs, Mel-RoFormer or AuK; word error on extracted speech.
