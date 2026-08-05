import bitsandbytes.nn as bnb_nn
import torch

orig_new = bnb_nn.Params4bit.__new__
def patched_new(cls, *args, **kwargs):
    print(f"Patched new called with kwargs: {list(kwargs.keys())}")
    kwargs.pop("_is_hf_initialized", None)
    return orig_new(cls, *args, **kwargs)

bnb_nn.Params4bit.__new__ = patched_new

try:
    p = bnb_nn.Params4bit(torch.randn(10), _is_hf_initialized=True)
    print("Success!")
except TypeError as e:
    print(f"Failed: {e}")
