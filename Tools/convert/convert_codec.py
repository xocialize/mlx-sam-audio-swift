"""mlx-sam-audio-swift V1 converter, codec half: the checkpoint's `audio_codec.*` (DAC-VAE with the AudioSeal watermark branch)
-> codec.safetensors in MLX layout.

  <oracle venv>/python convert_codec.py --out <bundle dir> [--repo facebook/sam-audio-base] [--dtype float32|float16]

Keys keep upstream's paths (minus `audio_codec.`). weight_norm is folded (`weight_g`, `weight_v` -> `weight`); Conv1d (O, I, K) ->
(O, K, I); ConvTranspose1d (I, O, K) -> (O, K, I); Snake alpha (1, C, 1) -> (C,); each LSTM layer's two biases are summed
into `bias_l{n}`.
"""
import argparse, pathlib, re
import numpy as np, torch
from huggingface_hub import snapshot_download
from safetensors.numpy import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--repo", default="facebook/sam-audio-base"); ap.add_argument("--dtype", default="float32")
a = ap.parse_args()
dt = {"float32": np.float32, "float16": np.float16}[a.dtype]
sd = torch.load(pathlib.Path(snapshot_download(a.repo)) / "checkpoint.pt", map_location="cpu", weights_only=True)
sd = {k[len("audio_codec."):]: v.double() for k, v in sd.items() if k.startswith("audio_codec.")}
TRANSPOSED = re.compile(r"^decoder\.model\.[1-9]\d*\.block\.(1|3)\.")   # the two ConvTranspose1d of each decoder block
out, n_wn, n_tr = {}, 0, 0
for k, v in sd.items():
    if k.endswith("weight_v") or k.endswith("bias_hh_l0") or k.endswith("bias_hh_l1"):
        continue
    if k.endswith("weight_g"):
        base = k[: -len("weight_g")]
        vv = sd[base + "weight_v"]
        w = v * vv / vv.flatten(1).norm(dim=1).reshape(-1, 1, 1)      # weight_norm, dim = 0
        n_wn += 1; k = base + "weight"; v = w
    if k.endswith(".alpha"):
        v = v.reshape(-1)
    elif re.search(r"lstm\.bias_ih_l(\d)$", k):
        n = k[-1]; v = v + sd[k.replace("bias_ih", "bias_hh")]; k = k.replace(f"bias_ih_l{n}", f"bias_l{n}")
    elif k.endswith(".weight") and v.ndim == 3:
        if TRANSPOSED.match(k):
            v = v.permute(1, 2, 0); n_tr += 1                           # (I, O, K) -> (O, K, I)
        else:
            v = v.permute(0, 2, 1)                                      # (O, I, K) -> (O, K, I)
    out[k] = np.ascontiguousarray(v.numpy().astype(dt))
save_file(out, str(pathlib.Path(a.out) / "codec.safetensors"), metadata={"format": "mlx", "source": a.repo})
print(len(out), "tensors", sum(v.size for v in out.values()) / 1e6, "M; weight_norm folded:", n_wn, "; transposed convs:", n_tr)
