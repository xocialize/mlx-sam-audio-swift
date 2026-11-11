# Tools

Both steps need access to the gated `facebook/sam-audio-base` repository and a Python environment with `torch`,
`transformers`, `safetensors`, `huggingface_hub` (the versions the goldens were captured with are in the audio area's
`WIP/sam-audio-port/oracle/requirements.lock`).

## convert

```
python Tools/convert/convert.py       --out <bundle dir> [--dtype float32|float16]
python Tools/convert/convert_codec.py --out <bundle dir> [--dtype float32]
```

Then save the `t5-base` tokenizer into the bundle (`AutoTokenizer.from_pretrained("t5-base").save_pretrained(<bundle dir>)`)
and copy the SAM `LICENSE` beside it. The bundle holds `model.safetensors`, `text_encoder.safetensors`,
`codec.safetensors`, `config.json` and the tokenizer files.

## oracle-capture

`npz_to_npy.py <goldens dir> <out dir>` turns the oracle's `.npz` captures into one `.npy` per tensor for
`samaudio-gates`. The captures themselves come from `capture.py` / `capture_decode.py` in the audio area's oracle
directory, which run upstream's own code on the gated checkpoint.

## Parity lanes

```
swift run -c release samaudio-gates all --bundle <fp32 bundle> --goldens <npy dir>
```

Lanes run at fp32 with MLX's TF32 matmul off (the tool sets `MLX_ENABLE_TF32=0` itself); with it on, the transformer
and text-encoder lanes read about 55 dB instead of 110–118 dB.
