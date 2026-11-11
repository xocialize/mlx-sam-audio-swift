# mlx-sam-audio-swift — porting spec

Meta's SAM-Audio base (text- and span-prompted source separation) on Swift-MLX. Task AB-T-0470; approval AB-D-0221;
licence AB-D-0220 (SAM License — `LICENSE-SAM` is the weights' licence text); watermark AB-D-0222 (embed it).

- **Upstream:** `facebookresearch/sam-audio` @ `bb4c699`, checkpoint `facebook/sam-audio-base` (gated), codec
  `facebookresearch/dacvae` @ `414c207`, text encoder `t5-base`.
- **Oracle:** `mlxengine-audio/WIP/sam-audio-port/oracle` (pinned venv, `capture.py`, `capture_decode.py`); goldens as
  one `.npy` per tensor via `Tools/oracle-capture/npz_to_npy.py`. Four cases: rain, speech over a song, drums in a
  trio, a cat with four `+` spans.
- **Scope:** the text + span path. Not ported: the vision encoder and visual prompts, the rankers, the judge, span
  prediction (PE-A-Frame).
- **Not optional on this path:** `align_masked_video`'s constant (see `SAMAudioModel.swift`); mlx-audio drops it
  (Neywa-Labs/mlx-audio#1176).
- **Lanes run at fp32 with `MLX_ENABLE_TF32=0`** (the gates tool sets it and prints what latched). Floor: 100 dB
  relative L2 against upstream PyTorch CPU fp32.

## Phases

| Phase | Gate | State |
|---|---|---|
| V1 converter | `Tools/convert`: `model.safetensors` (318 tensors, 1,150.3 M), `text_encoder.safetensors` (99 tensors, 109.6 M), `codec.safetensors` (251 tensors, 107.6 M; weight_norm folded), `config.json`, tokenizer | done 2026-11-11 |
| S0 key contract | module parameter paths == bundle keys, both files, 0 missing / 0 unused | **PASSED 2026-11-11** (`samaudio-gates keys`) |
| S1 text encoder | T5 features against upstream, 4 cases | **PASSED 2026-11-11**: 116.7–118.5 dB |
| S1 velocity | one forward at t = 0.3 on golden inputs, 4 cases (one with span anchors) | **PASSED 2026-11-11**: 110.7–112.3 dB |
| S2 solve | 16 midpoint steps from the golden noise, 4 cases | **PASSED 2026-11-11**: 115.6–117.6 dB; 1.6–2.0 s per solve (fp32, release, M5 Max) |
| S1 codec encode | waveform → features | **PASSED 2026-11-11**: 94.7–113.7 dB (floor 90: the Python-MLX rung's own worst case is 94.5) |
| S1 codec decode | decoder blocks, plain waveform, and the watermarked waveform with the fixed message | **PASSED 2026-11-11**: blocks 116.5–122.7, plain and marked 116.7–121.4 dB; the watermark signal alone 92.6–103.3 dB |
| S5 windowed decode | windows of 25 latent frames + 12 of context against the single pass | **PASSED 2026-11-11**: identical to the bit; 10 s decode + watermark 3.8–5.5 GB instead of ~17 GB |
| S2 end to end | wav + prompt (+ spans) → marked target / residual, golden noise, fixed message | **PASSED 2026-11-11**: tokens and anchors equal; 97.0–110.7 dB |
| S2b tokenizer + gates | swift-transformers tokenizer; the pre-registered V0 gates through `samaudio-gates batch` | **PASSED 2026-11-11** at fp32 and fp16 (MEASUREMENTS.md) |
| S6 tiers | fp16 DiT (T5 and codec fp32) on the V0 mixtures and gates | **PASSED 2026-11-11**: same medians as fp32, 4.48 GB after load / 8.60 GB peak, 4.44 s per 10 s clip. int8 not tried |
| V3 engine package | `MLXSAMAudio` on `audioSeparation` (engine 0.77.0, contract 1.62.0: described target + spans), conformance, footprint from `phys_footprint` | **PASSED 2026-11-11**: 22 offline tests; `validate` through MLXServeEngine twice (explicit bundle; engine download); declared 3.5 + 6.0 GB; clips 1–30 s (5–30 s gated). Speed through the engine not established (GPU contention) |
| V4 publish | repo + bundles under the SAM License | code: github.com/xocialize/mlx-sam-audio-swift `v0.1.0`; weights: `mlx-community/sam-audio-base-swift-fp16` @ `2cf624a4e250` (**published 2026-11-11**, hashes verified from a fresh download). Engine-package pins follow V3 |
| V5 Gaea plugin | needs the MES change (AB-A-0269) | pending |

## Notes for whoever continues

- `Attention.splitHeads` follows upstream's `reshape_heads`: the head index is the projection's FAST axis. The lanes
  above would read ~0 dB with the usual heads-major split.
- Prompts are lowercase noun / verb phrases ("speech", "dog barking"); sentences fail (upstream README; V0c).
- Guitar-like stems do not separate (1.3 dB); do not claim them.
- The watermark branch must be ported from `dacvae` directly: mlx-audio's `decode_with_watermark` does not run.
- Norms run in fp32 inside a half-precision DiT (upstream does the same); without it fp16 separates nothing.
- The decode is windowed (25 frames + 12 of context) and bit-identical to a single pass; do not widen the core without
  re-running the `codec` lane's "windowed vs single pass" row.
- Licence: the package is under the SAM License (`NOTICE.md`): the model code is a translation of SAM-licensed code.
