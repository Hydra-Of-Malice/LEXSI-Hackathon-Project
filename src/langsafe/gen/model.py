"""Load Tiny Aya (full precision, NF4, or with a LoRA/steering adapter) the same way for every arm."""
from __future__ import annotations


def load(model_id: str, dtype: str = "float16", quant: str | None = None, adapter: str | None = None,
         device_map: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"device_map": device_map, "dtype": getattr(torch, dtype)}
    if quant == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=getattr(torch, dtype),
                                                           bnb_4bit_use_double_quant=False)
    elif quant:
        raise ValueError(f"unsupported in-process quantisation {quant!r} (GGUF runs through llama.cpp, see quant/)")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok
