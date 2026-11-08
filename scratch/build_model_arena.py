import os, json

os.makedirs('model_arena', exist_ok=True)

# 1. kernel-metadata.json
metadata = {
  "id": "baonguyenduyquoc/name-that-llm-model-arena-benchmark",
  "title": "Name That LLM Model Arena Benchmark",
  "code_file": "arena-benchmark.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "keywords": [
    "gpu"
  ],
  "dataset_sources": [
    "dhruvtibarewal/nlm-matched-clean",
    "dhruvtibarewal/nlm-enc-v5",
    "dhruvtibarewal/nlm-l24-adapters-v5",
    "alexchilton/ntl-ts-gemma2-9b",
    "seyitkaangunes/ntl-extra-api-passages",
    "abdullah1afify/ntl-generations",
    "dophuchung/ntl-proxy-generations",
    "nishantdahal/ntl-handwriting-blend",
    "pjmathematician/ultrabert-380m-ntl"
  ],
  "kernel_sources": [],
  "competition_sources": [
    "name-that-llm"
  ],
  "model_sources": [
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen-3/Transformers/14b-base/1",
    "google/gemma-2/Transformers/9b/2"
  ],
  "machine_shape": "NvidiaTeslaT4"
}

with open('model_arena/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("model_arena/kernel-metadata.json written successfully.")
