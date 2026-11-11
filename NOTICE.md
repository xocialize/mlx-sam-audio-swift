# Licence and notices

**This package is distributed under the SAM License** (`LICENSE`, Meta, last updated November 19, 2025).

Why not MIT, like the fleet's other ports: `SAMAudioCore`'s model code is a translation of Meta's
`facebookresearch/sam-audio` (commit `bb4c699`), whose code and weights are both "SAM Materials". §1b-i of the SAM
License allows derivative works to be distributed only under the same agreement, with a copy of it. The weights this
package loads (`facebook/sam-audio-base`, converted) are under the same licence.

What the SAM License asks of anyone using or redistributing this package or its weights (read the `LICENSE`; this is a
summary, not a substitute):
- Redistribution, including of modified or converted versions, only under the SAM License, with a copy of it.
- No use for ITAR-controlled activities or end uses prohibited by trade controls, "including those related to
  military or warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or
  illegal weapons"; compliance with sanctions and applicable law.
- Not to "involve or encourage others to reverse engineer, decompile or discover the underlying components" of the
  SAM Materials.
- Acknowledge the SAM Materials in research publications.
- You indemnify Meta against third-party claims arising from your use or distribution (§5b). No warranty.
- Meta may amend the licence (§8).

Commercial use is permitted.

**Outputs are watermarked.** As in upstream, the codec's decoder embeds an AudioSeal watermark in every waveform it
produces (a random 16-bit message per decode unless one is supplied). There is no option to turn it off.

## Third-party components

| Component | Origin | Licence | Use |
|---|---|---|---|
| SAM-Audio model code and weights | `facebookresearch/sam-audio` @ `bb4c699`; `facebook/sam-audio-base` | SAM License | translated (DiT, anchors, alignment, pipeline); weights converted by `Tools/convert` |
| DAC-VAE codec and watermark | `facebookresearch/dacvae` @ `414c207`; weights inside the SAM-Audio checkpoint | code Apache-2.0; weights SAM License | translated (`Codec/DACVAE.swift`) |
| T5-base encoder | `t5-base` (Google), via `transformers` | Apache-2.0 | architecture translated; encoder weights and tokenizer converted |
| mlx-swift | `ml-explore/mlx-swift` | MIT | runtime |
| swift-transformers | `huggingface/swift-transformers` | Apache-2.0 | tokenizer |
| NPY / WeightIO / TF32 helpers | fleet ports (mlx-heartmula-swift and predecessors) | MIT | lifted in shape |

Not included and not used: the vision encoder (visual prompts), ImageBind and CLAP rankers, the SAM-Audio judge, and
the PE-A-Frame span predictor.
