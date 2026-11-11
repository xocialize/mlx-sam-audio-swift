"""mlx-sam-audio-swift V1 converter: the gated upstream checkpoint -> an MLX-layout bundle whose keys are the Swift module paths.

  <oracle venv>/python convert.py --out <bundle dir> [--repo facebook/sam-audio-base] [--dtype float32|float16]

Writes:
  model.safetensors         DiT + input projections + anchors + the align_masked_video terms the text path needs
                            (gate, conv.bias, layer_norm.*; conv.weight is only for real video features and is dropped).
                            torch key names; Conv1d weights (O, I, K) -> (O, K, I).
  text_encoder.safetensors  the T5-base ENCODER from `t5-base` (Apache-2.0), compact keys, always fp32.
  config.json               the upstream config's text path + provenance.
The codec (codec.safetensors) is written by convert_codec.py.
"""
import argparse, json, pathlib
import numpy as np, torch
from huggingface_hub import snapshot_download
from safetensors.numpy import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--repo", default="facebook/sam-audio-base"); ap.add_argument("--dtype", default="float32")
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
dt = {"float32": np.float32, "float16": np.float16}[a.dtype]
snap = pathlib.Path(snapshot_download(a.repo))
sd = torch.load(snap / "checkpoint.pt", map_location="cpu", weights_only=True)

model = {}
for k, v in sd.items():
    if k.startswith(("vision_encoder.", "audio_codec.")) or k == "align_masked_video.conv.weight":
        continue
    x = v.numpy()
    if k.endswith(".project.weight"):          # x_embedder Conv1d (O, I, K) -> (O, K, I)
        assert x.ndim == 3; x = x.transpose(0, 2, 1)
    model[k] = np.ascontiguousarray(x.astype(dt))
save_file(model, str(out / "model.safetensors"), metadata={"format": "mlx", "source": a.repo})

from transformers import T5EncoderModel
t5 = T5EncoderModel.from_pretrained("t5-base").state_dict()
enc = {"shared.weight": t5["shared.weight"], "final_norm.weight": t5["encoder.final_layer_norm.weight"],
       "relative_attention_bias.weight": t5["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]}
n = 1 + max(int(k.split(".")[2]) for k in t5 if k.startswith("encoder.block."))
for i in range(n):
    p = f"encoder.block.{i}.layer"
    for s in "qkvo":
        enc[f"blocks.{i}.{s}.weight"] = t5[f"{p}.0.SelfAttention.{s}.weight"]
    enc[f"blocks.{i}.attn_norm.weight"] = t5[f"{p}.0.layer_norm.weight"]
    enc[f"blocks.{i}.wi.weight"] = t5[f"{p}.1.DenseReluDense.wi.weight"]
    enc[f"blocks.{i}.wo.weight"] = t5[f"{p}.1.DenseReluDense.wo.weight"]
    enc[f"blocks.{i}.ff_norm.weight"] = t5[f"{p}.1.layer_norm.weight"]
used = 3 + n * 8
assert used == len({k for k in t5 if not k.startswith("encoder.embed_tokens")}), (used, len(t5))
# The text encoder is always fp32: T5 overflows in half precision, and the pipeline loads it as fp32 whatever the DiT's dtype.
save_file({k: np.ascontiguousarray(v.numpy().astype(np.float32)) for k, v in enc.items()}, str(out / "text_encoder.safetensors"), metadata={"format": "mlx", "source": "t5-base"})

cfg = json.loads((snap / "config.json").read_text())
keep = {k: cfg[k] for k in ("in_channels", "audio_codec", "text_encoder", "transformer", "num_anchors", "anchor_embedding_dim")}
keep["text_encoder"] |= {"layers": n, "heads": 12, "d_kv": 64, "d_ff": 3072, "vocab": int(t5["shared.weight"].shape[0]), "relative_buckets": 32, "relative_max_distance": 128}
keep["provenance"] = {"source": a.repo, "source_revision": snap.name, "dtype": a.dtype, "text_encoder_dtype": "float32", "codec_dtype": "float32", "upstream_code": "facebookresearch/sam-audio@bb4c699", "license": "SAM License (see LICENSE)"}
(out / "config.json").write_text(json.dumps(keep, indent=2))
print("model", len(model), "tensors", sum(v.size for v in model.values()) / 1e6, "M | t5", len(enc), "tensors", sum(v.numel() for v in enc.values()) / 1e6, "M")
