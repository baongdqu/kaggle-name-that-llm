import os, json, shutil

# Create directory structure for v77, v78, v79, v80
for v in ['v77', 'v78', 'v79', 'v80']:
    d = f"champion_{v}"
    os.makedirs(d, exist_ok=True)
    print(f"Directory {d} ready.")

