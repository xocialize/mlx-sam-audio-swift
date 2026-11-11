"""Goldens for the Swift lanes: WIP/sam-audio-port/oracle/goldens/*.npz (capture.py, capture_decode.py) -> one .npy per tensor.
  <oracle venv>/python npz_to_npy.py <goldens dir> <out dir>
Adds the T5 token ids for each case's prompt (the tokenizer upstream uses)."""
import pathlib, sys, numpy as np
from transformers import AutoTokenizer
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
tok = AutoTokenizer.from_pretrained("t5-base")
for f in sorted(src.glob("*.npz")):
    g = np.load(f, allow_pickle=True); d = dst / f.stem.replace("-decode", ""); d.mkdir(parents=True, exist_ok=True)
    for k in g.files:
        v = g[k]
        if k == "prompt":
            ids = tok([str(v)], truncation=True, max_length=512, padding="longest", return_tensors="np")["input_ids"].astype(np.int32)
            np.save(d / "input_ids.npy", ids); (d / "prompt.txt").write_text(str(v)); continue
        if v.dtype == np.int64: v = v.astype(np.int32)
        if v.dtype == np.float64: v = v.astype(np.float32)
        np.save(d / f"{k}.npy", v)
    print(d.name, sorted(p.name for p in d.iterdir())[:40])
