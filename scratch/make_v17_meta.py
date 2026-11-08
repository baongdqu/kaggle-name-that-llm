import json
import os
import ast

os.makedirs('champion_v17', exist_ok=True)

# 1. Metadata for V17 Apex Crown
meta_v17 = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v17-apex-crown",
  "title": "Name That LLM Champion V17 Apex Crown",
  "code_file": "champion-v17.ipynb",
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
    "seyitkaangunes/ntl-extra-api-passages",
    "zbigniewkirchner/ntl-extra-gpt-passages",
    "abdullah1afify/ntl-generations",
    "dophuchung/ntl-proxy-generations",
    "zbigniewkirchner/ntl-qwen3-8b-lora-instyle",
    "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
    "lucasboesen/ntl-free-proxy-generations",
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
    "bbucxi/qwen35-9b-safetensors",
    "pjmathematician/ultrabert-380m-ntl",
    "nishantdahal/ntl-two-stage-lora",
    "lightningv08/ntl-olmo2-native24-final-inference-assets"
  ],
  "kernel_sources": [
    "lucasboesen/name-that-llm-glm4-9b-trunc25",
    "lucasboesen/name-that-llm-extract-encoders2",
    "lucasboesen/name-that-llm-electra-large-weights",
    "lucasboesen/name-that-llm-desklib-weights",
    "lightningv08/ntl-lora24-fivefold-training",
    "lucasboesen/name-that-llm-train-lora24-s7",
    "lucasboesen/name-that-llm-train-tokattn",
    "lucasboesen/name-that-llm-train-tokattn2",
    "vnton37nguyn/ntl-train-lora-qwen25-instyle",
    "lightningv08/olmo2-7b-fp16-offline-weights"
  ],
  "competition_sources": [
    "name-that-llm"
  ],
  "model_sources": [
    "qwen-lm/qwen-3/Transformers/14b-base/1",
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen2.5/Transformers/7b/1"
  ],
  "machine_shape": "NvidiaTeslaT4"
}

with open('champion_v17/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta_v17, f, indent=2)

print("Saved champion_v17/kernel-metadata.json")
