import struct, json, re

path = "scratch/gemma_meta/adapter_model.safetensors"
with open(path, "rb") as f:
    length = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(length).decode("utf-8"))

layer_indices = set()
for k in header.keys():
    m = re.search(r"layers\.(\d+)\.", k)
    if m:
        layer_indices.add(int(m.group(1)))

print("Total layer indices found:", sorted(list(layer_indices)))
