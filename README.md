# mlx-sam-audio-swift

Meta's [SAM-Audio](https://github.com/facebookresearch/sam-audio) on Apple Silicon with Swift-MLX: describe a sound in
a few words and get it separated from a mixture, together with everything else.

- **In:** a mono 48 kHz mixture, a short description ("speech", "dog barking", "drums"), optional time spans where the
  sound is or is not present.
- **Out:** `target` (the described sound) and `residual` (the rest), 48 kHz mono, the mixture's length. Both are
  generated, not filtered: they are not sample-aligned with the mixture, and they carry the codec's watermark.
- **Model:** SAM-Audio base (1.26 B parameters): T5-base text encoder, a flow-matching transformer, the DAC-VAE codec.

**Licence: the SAM License**, for the code and the weights. Commercial use is allowed; there are use restrictions and
redistribution conditions. Read [`NOTICE.md`](NOTICE.md) and [`LICENSE`](LICENSE).

## Status

`SAMAudioCore` (the model) and `samaudio-gates` (the parity tool) are complete and gated against upstream PyTorch; see
[`PORTING-SPEC.md`](PORTING-SPEC.md) and [`MEASUREMENTS.md`](MEASUREMENTS.md). The MLXEngine package and published
weight bundles are not in this release.

## Use

```swift
import SAMAudioCore

let pipeline = try await SAMAudioPipeline(bundle: bundleURL, modelDType: .float16)
let out = try pipeline.separate(wav: mixture, prompt: "speech", seed: 7)   // MLXArray (samples,), 48 kHz mono
// out.target, out.residual
```

The bundle is made from the gated upstream checkpoint with `Tools/convert` (see `Tools/README.md`).

## What to expect

- **Write prompts as lowercase noun or verb phrases** — "speech", "rain", "door knock" — not sentences. The model was
  trained on that form; "a person speaking" misses where "speech" works.
- Works well on general sounds, speech against music, music against speech, drums and bass.
- **Does not separate guitar-like stems** in our tests.
- Spans help intermittent sounds (a cat, a door knock). A span that covers the whole clip can hurt steady sounds.
- One 10 s clip per call; longer audio needs windowing by the caller in this release.

## This port and mlx-audio's

mlx-audio's Python port of SAM-Audio omits a term upstream applies on every call (`align_masked_video` on a zero video
feature) and is 19–25 dB from upstream per model step because of it
([mlx-audio#1176](https://github.com/Neywa-Labs/mlx-audio/issues/1176)). The `mlx-community/sam-audio-*` bundles made
for it do not contain the tensors this package needs; convert from the upstream checkpoint.
