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
[`PORTING-SPEC.md`](PORTING-SPEC.md) and [`MEASUREMENTS.md`](MEASUREMENTS.md). `MLXSAMAudio` is the MLXEngine package on
`audioSeparation` (engine ≥ 0.77.0): a described target, optional time spans, `target` and `residual` stems; mixtures
of 1–30 s at any sample rate; 3.5 GB resident + 6.0 GB activation.

**Weights:** [`mlx-community/sam-audio-base-swift-fp16`](https://huggingface.co/mlx-community/sam-audio-base-swift-fp16)
(3.2 GB; fp16 transformer, fp32 T5 encoder and codec; SAM License), verified at revision `2cf624a4e250`.

## Use

```swift
// Package.swift: .package(url: "https://github.com/xocialize/mlx-sam-audio-swift", from: "0.1.0") → product "SAMAudioCore"
import SAMAudioCore

let pipeline = try await SAMAudioPipeline(bundle: bundleURL)   // a local copy of mlx-community/sam-audio-base-swift-fp16
let out = try pipeline.separate(wav: mixture, prompt: "speech", seed: 7)   // MLXArray (samples,), 48 kHz mono
// out.target, out.residual
```

Download the bundle with `hf download mlx-community/sam-audio-base-swift-fp16 --local-dir <dir>`, or make it from the
gated upstream checkpoint with `Tools/convert` (see `Tools/README.md`).

Through MLXEngine:

```swift
// product "MLXSAMAudio"
let engine = MLXServeEngine()
await engine.useModelStore(ModelStore(root: modelsRoot))
let id = try await engine.register(SAMAudioPackage.registration, configuration: SAMAudioConfiguration(), id: "sam-audio-base")
try await engine.prepare(.audioSeparation, package: id)          // downloads mlx-community/sam-audio-base-swift-fp16 if needed
let out = try await engine.run(AudioSeparationRequest(audio: mixture, targetDescription: "speech",
                                                      spans: [], metaData: ["seed": .int(7)]),
                               package: id) as! AudioSeparationResponse
// out.stems[.target], out.stems[.residual]: 48 kHz mono .wav, the mixture's length
```

## What to expect

- **Write prompts as lowercase noun or verb phrases** — "speech", "rain", "door knock" — not sentences. The model was
  trained on that form; "a person speaking" misses where "speech" works.
- Works well on general sounds, speech against music, music against speech, drums and bass.
- **Does not separate guitar-like stems** in our tests.
- Spans help intermittent sounds (a cat, a door knock). A span that covers the whole clip can hurt steady sounds.
- Mixtures of 5–30 s separate reliably; under 5 s they do not (2 s clips fail a quarter of the time). Longer audio
  needs windowing by the caller in this release.

## This port and mlx-audio's

mlx-audio's Python port of SAM-Audio omits a term upstream applies on every call (`align_masked_video` on a zero video
feature) and is 19–25 dB from upstream per model step because of it
([mlx-audio#1176](https://github.com/Neywa-Labs/mlx-audio/issues/1176)). The `mlx-community/sam-audio-*` bundles made
for it do not contain the tensors this package needs; use the bundle above.
