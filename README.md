# Apertus-8B-Instruct-2509 Setup

This directory contains a ready-to-use setup for the **Apertus-8B-Instruct-2509** model by Swiss AI Initiative.

## Contents
- `run_apertus_8b.py`: A Python script to load the model and run a sample inference.
- Installed libraries: `transformers`, `torch`, `accelerate`, `bitsandbytes`.

## How to use

To run the model and see a sample response, execute:

```bash
python3 run_apertus_8b.py
```

## Model Details
- **Model Name:** `swiss-ai/Apertus-8B-Instruct-2509`
- **Architecture:** Decoder-only transformer
- **Features:** 1811 natively supported languages, 65k context length.
- **Hardware Requirement:** ~16GB VRAM (Full BF16) or ~5GB VRAM (4-bit).

## Troubleshooting
- **Memory:** The model is currently configured to load in `bfloat16`. If you run into Out-Of-Memory (OOM) errors on smaller GPUs, consider using 4-bit quantization by adding `load_in_4bit=True` to the `AutoModelForCausalLM.from_pretrained` call.
- **xIELU:** The model uses a new activation function. A Python fallback is used if the CUDA-fused version is unavailable.
