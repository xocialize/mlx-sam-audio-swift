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

| **published bundle, as stored** (`mlx-community/sam-audio-base-swift-fp16`) | same rows, per-row scores equal to the fp16 line (max difference 0.000 dB) | same | same | 3.0–3.8 s, median 3.77 s | **3.34 / 7.47 GB** |

The fp32 and fp16 lines load the fp32 conversion (the fp16 line casts at load, which leaves more resident); the last
line is the shipped bundle. Fresh-download check: 7 of 7 file hashes equal to the local bundle, and separations from
the download agree with the pre-upload run at 48–59 dB (the difference is the random watermark message).

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

## V3 — the engine package (`MLXSAMAudio`, engine 0.77.0 / contract 1.62.0)

**Clip lengths** (`WIP/sam-audio-eval/MEASUREMENTS.md` §V3 length sweep; rule written first: claimable = median gain ≥ 6 dB
and at most 1 of 8 worse than the mixture). Four 30 s speech + steady-effect mixtures cropped to each length, both
targets, one draw, the published bundle:

| Length | Median gain | Worst | Worse than the mixture | Per clip | `phys_footprint` peak |
|---|---|---|---|---|---|
| 2 s | 9.86 dB | −5.95 dB | **2 of 8 — not claimable** | 0.81 s | 7.46 GB |
| 5 s | 11.23 dB | 8.87 dB | 0 | 1.92 s | 7.46 GB |
| 7.777 s (not a hop multiple) | 11.15 dB | 0.38 dB | 0 | 2.90 s | 7.64 GB |
| 10 s | 12.11 dB | 8.60 dB | 0 | 3.69 s | 7.64 GB |
| 20 s | 11.51 dB | 8.06 dB | 0 | 7.25 s | 9.76 GB |
| 30 s | 10.44 dB | 7.30 dB | 0 | 15.38 s | 12.92 GB |

Output length equals the input's at every length. The surface takes 1–30 s and says mixtures under 5 s are unreliable.
(These peaks are the CLI's, with MLX's default cache; the engine runs below are with the package's 1 GiB cache limit.)

**Live through `MLXServeEngine`** (`samaudio-gates validate`, two runs: an explicit bundle, and an empty models folder
the engine filled itself — 3.0 GB in 76 s): registered under the default licence policy with **no advisory**; five
separations (default stems, a re-run with untrimmed mixed-case text, target only, spans, a 30 s mixture) each return
the asked stems at the mixture's exact length, 48 kHz mono, non-silent; **8 of 8 refusals** (no description, blank,
a fixed stem beside a description, a speaker pick, a span ending before it starts, a span past the mixture's end, a
60 s mixture, a 201-character description); a 30 s request cancelled mid-run stops 0.2–1.7 s after the cancel with
`CancellationError`, and the next request runs; evict returns the process to 1.5–2.2 GB.

| | run 1 | run 2 |
|---|---|---|
| `phys_footprint` after `prepare` | 3.36 GB | 3.38 GB |
| lifetime peak, requests ≤ 10 s | 6.28 GB | 6.45 GB |
| lifetime peak with the 30 s request | 9.01 GB | 9.07 GB |

**Declared: 3.5 GB resident + 6.0 GB peak activation** (9.5 GB ≥ 9.07). By the engine's 0.7 × RAM admission that is a
16 GB-Mac tier.

**Offline conformance:** 22 tests pass (`swift test`): manifest (C0 / C1 / C7 / C8 with the port-code notice /
C-memory / C11 / provenance), MAT-1..5, CAN-1..3, the request plane, audio IO.

**Speed through the engine is NOT established.** The validate runs were taken while the machine's GPU read 99 %
utilised by something outside this work: 10 s requests took 3.5–4.4 s in one run and 18–63 s in two others, and a
direct `batch` re-run at the same time read 21–27 s where the quiet runs above read 3.0–3.8 s. TF32 on or off made no
difference (22.6 s against 24.9 s in that state). The quiet-machine figure remains 3.77 s per 10 s clip; it needs
re-measuring through the engine on an idle machine.

**Found on the way:** `AVAudioFile.read(into:)` returned a 480 000-frame float WAV 788 frames short in one call (the
stems came back short and a valid span was refused as past the end); the mixture is now read in a loop until the
file's length is reached. The same single-call read is in other fleet wrappers.

## Not measured

int8 and smaller tiers; clips longer or shorter than 10 s; long-form windowing of the whole separation; the small and
large checkpoints; a head-to-head with Demucs, Mel-RoFormer or AuK; word error on extracted speech.
