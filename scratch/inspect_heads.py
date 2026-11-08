import struct, json

path = "scratch/gemma_meta/heads.safetensors"
with open(path, "rb") as f:
    length = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(length).decode("utf-8"))

print("heads.safetensors keys:")
for k, v in header.items():
    if k != "__metadata__":
        print(f"  {k:20s}: {v.get('shape')} {v.get('dtype')}")
