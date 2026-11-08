import subprocess, time, sys

print("Tracking submission 57973975 (Champion V74)...")

for i in range(50):
    res = subprocess.run(["kaggle", "competitions", "submissions", "-c", "name-that-llm"], capture_output=True, text=True)
    lines = res.stdout.split('\n')
    v74_line = None
    for l in lines:
        if "57973975" in l:
            v74_line = l
            break
    if v74_line:
        print(f"[{i*15:3d}s] {v74_line}")
        if "COMPLETE" in v74_line:
            print("SUBMISSION SCORED SUCCESSFULLY!")
            break
        if "ERROR" in v74_line:
            print("SUBMISSION FAILED!")
            break
    else:
        print(f"[{i*15:3d}s] Submission not found yet in output.")
    time.sleep(15)

