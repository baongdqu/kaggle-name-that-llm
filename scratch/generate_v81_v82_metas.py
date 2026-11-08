import json

meta_template = {
  "id": "baonguyenduyquoc/ntl-champion-{version}-{slug}",
  "title": "NTL Champion {version_upper} {title_slug}",
  "code_file": "champion-{version}.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "dataset_sources": [
    "seyitkaangunes/ntl-extra-api-passages",
    "abdullah1afify/ntl-generations",
    "dophuchung/ntl-proxy-generations",
    "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
    "dhruvtibarewal/nlm-matched-clean",
    "josephadamski91/ntl-atlas-qwen3-8b-v1",
    "lucasboesen/ntl-free-proxy-generations",
    "dhruvtibarewal/nlm-enc-v5",
    "nishantdahal/ntl-handwriting-blend",
    "dhruvtibarewal/nlm-l24-adapters-v5",
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
    "pjmathematician/ultrabert-380m-ntl",
    "ilyass323/llama-3-1-8b-bf16",
    "crtproduction/name-that-llm-open-pool"
  ],
  "kernel_sources": [],
  "competition_sources": [
    "name-that-llm"
  ],
  "model_sources": [
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen-3/Transformers/14b-base/1",
    "qwen-lm/qwen-3/Transformers/8b/1",
    "qwen-lm/qwen-3/Transformers/14b/1",
    "qwen-lm/qwen2.5/Transformers/7b/1",
    "qwen-lm/qwen2.5/Transformers/14b/1"
  ]
}

configs = {
    "v81": ("internal-hungarian-apex", "Internal Hungarian Apex", "v81"),
    "v82": ("oracle-global-unconstrained", "Oracle Global Unconstrained", "v82")
}

for v, (slug, title_slug, ver) in configs.items():
    m = dict(meta_template)
    m["id"] = m["id"].format(version=v, slug=slug)
    m["title"] = m["title"].format(version_upper=v.upper(), title_slug=title_slug)
    m["code_file"] = f"champion-{v}.ipynb"
    with open(f"champion_{v}/kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(f"Generated metadata for {v}")

