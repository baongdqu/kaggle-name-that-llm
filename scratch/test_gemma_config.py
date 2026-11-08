from transformers import AutoConfig, AutoTokenizer

# Test Gemma2 config manipulation
config_name = "google/gemma-2-9b"
print(f"Testing Gemma2 config loading for {config_name}...")
try:
    cfg = AutoConfig.from_pretrained(config_name)
    print("Default num_hidden_layers:", cfg.num_hidden_layers)
    cfg.num_hidden_layers = 24
    print("Modified num_hidden_layers:", cfg.num_hidden_layers)
except Exception as e:
    print("Config error (offline):", repr(e))
