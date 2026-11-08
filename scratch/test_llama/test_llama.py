import os

print("Top level /kaggle/input contents:")
print(os.listdir("/kaggle/input"))

for root, dirs, files in os.walk("/kaggle/input"):
    print(f"DIR: {root} -> DIRS: {dirs} -> FILES: {files[:5]}")
