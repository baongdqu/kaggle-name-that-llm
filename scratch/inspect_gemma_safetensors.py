import struct, json

# Read safetensors header without needing third-party libraries
path = "scratch/gemma_meta/adapter_model.safetensors"
try:
    with open(path, "rb") as f:
        length_bytes = f.read(8)
        length = struct.unpack("<Q", length_bytes)[0]
        header_bytes = f.read(length)
        header = json.loads(header_bytes.decode("utf-8"))
    
    print(f"Successfully loaded safetensors header! Total keys: {len(header)}")
    print("Sample keys:")
    keys = list(header.keys())
    for k in keys[:15]:
        print("  ", k, header[k].get("shape"))
except Exception as e:
    print("Error reading header:", repr(e))
