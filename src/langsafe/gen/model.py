"""Load Tiny Aya (full precision, NF4, or with a LoRA/steering adapter) the same way for every arm.

Runs on CUDA (Kaggle/Colab T4, RTX laptops), Apple Silicon (MPS) or CPU. NF4 needs CUDA (bitsandbytes);
on a Mac the 4-bit check uses GGUF through llama.cpp instead (see quant/gguf.py).
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_model(model_id: str) -> str:
    """Use a local copy when `LANGSAFE_MODELS` points at a folder holding e.g. `tiny-aya-fire/` (safetensors, as
    downloaded with `hf download CohereLabs/tiny-aya-fire --local-dir ...`). Otherwise the Hub id (and HF cache)."""
    root = os.environ.get("LANGSAFE_MODELS")
    if root:
        cand = Path(root).expanduser() / model_id.split("/")[-1]
        if (cand / "config.json").is_file():
            return str(cand)
    return model_id


def best_device() -> str:
    """`LANGSAFE_DEVICE` wins; otherwise cuda, then mps, then cpu."""
    import torch

    if os.environ.get("LANGSAFE_DEVICE"):
        return os.environ["LANGSAFE_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model_id: str, dtype: str = "float16", quant: str | None = None, adapter: str | None = None,
         device_map: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = best_device()
    if device_map is None:
        device_map = "auto" if dev == "cuda" else {"": dev}
    if dev == "cpu" and dtype == "float16":
        dtype = "float32"          # fp16 matmuls are slow or unsupported on CPU
    kwargs = {"device_map": device_map, "dtype": getattr(torch, dtype)}
    if quant == "nf4":
        if dev != "cuda":
            raise RuntimeError("NF4 (bitsandbytes) needs a CUDA GPU; on a Mac use the GGUF check (quant/gguf.py)")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=getattr(torch, dtype),
                                                           bnb_4bit_use_double_quant=False)
    elif quant:
        raise ValueError(f"unsupported in-process quantisation {quant!r} (GGUF runs through llama.cpp, see quant/)")
    path = resolve_model(model_id)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok
