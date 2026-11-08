import json
import os
import shutil

# Directories
os.makedirs('champion_v7', exist_ok=True)
os.makedirs('champion_v8', exist_ok=True)
os.makedirs('champion_v9', exist_ok=True)

# Common datasets and kernels metadata
datasets = [
    "bbucxi/qwen35-9b-safetensors",
    "pjmathematician/ultrabert-380m-ntl",
    "seyitkaangunes/ntl-extra-api-passages",
    "olegzholobov/name-that-llm-response-corpus"
]
kernels = [
    "lucasboesen/name-that-llm-glm4-9b-trunc25",
    "lucasboesen/name-that-llm-extract-encoders2",
    "lucasboesen/name-that-llm-electra-large-weights",
    "lucasboesen/name-that-llm-desklib-weights",
    "lightningv08/ntl-lora24-fivefold-training",
    "lucasboesen/name-that-llm-train-lora24-s7",
    "lucasboesen/name-that-llm-train-tokattn",
    "lucasboesen/name-that-llm-train-tokattn2"
]
models = [
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen2.5/Transformers/7b/1"
]

def make_meta(slug, title, notebook_name):
    return {
        "id": f"baonguyenduyquoc/{slug}",
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": datasets,
        "kernel_sources": kernels,
        "competition_sources": ["name-that-llm"],
        "model_sources": models,
        "machine_shape": "NvidiaTeslaT4"
    }

with open('champion_v7/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(make_meta('name-that-llm-champion-v7-zscore-quadgate', 'Name That LLM - Champion V7 ZScore QuadGate', 'champion-v7.ipynb'), f, indent=2)

with open('champion_v8/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(make_meta('name-that-llm-champion-v8-presinkhorn-quadgate', 'Name That LLM - Champion V8 PreSinkhorn QuadGate', 'champion-v8.ipynb'), f, indent=2)

with open('champion_v9/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(make_meta('name-that-llm-champion-v9-temp-sinkhorn', 'Name That LLM - Champion V9 Temp Annealed Sinkhorn', 'champion-v9.ipynb'), f, indent=2)

print("Metadata for V7, V8, V9 created successfully!")
