import glob
import os
import re

def check_log(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return f"read error: {e}"
        
    times = re.findall(r'total (\d+)s', content)
    if times:
        return f"total {times[-1]}s ({round(int(times[-1])/60, 1)}m)"
        
    json_times = re.findall(r'"time":\s*(\d+\.?\d*)', content)
    if json_times:
        max_t = float(json_times[-1])
        return f"last stream time {round(max_t, 1)}s ({round(max_t/60, 1)}m)"
        
    return "no time found"

print("--- Output folder logs ---")
for folder in sorted(glob.glob('v*_output')):
    for l in glob.glob(os.path.join(folder, '*.log')):
        print(f"{l}: {check_log(l)}")

print("\n--- Root logs ---")
for l in sorted(glob.glob('*.log')):
    print(f"{l}: {check_log(l)}")

print("\n--- Champion folders check ---")
for folder in sorted(glob.glob('champion_v*')):
    for l in glob.glob(os.path.join(folder, '*.log')):
        print(f"{l}: {check_log(l)}")

# Let's inspect transcript for submission scoring completion times
transcript_path = r"C:\Users\s3cr3t\.gemini\antigravity-ide\brain\f5ca0093-20f3-4308-a03b-7ccba8e6503a\.system_generated\logs\transcript.jsonl"
if os.path.exists(transcript_path):
    print("\n--- Scanning transcript for submission status updates ---")
    import json
    with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            if "SubmissionStatus.COMPLETE" in line and "57" in line:
                # Find submission ref and timestamp if possible
                try:
                    data = json.loads(line)
                    content = str(data.get("content", ""))
                    # extract ref and date
                    matches = re.findall(r'(57\d{6})\s+submission\.parquet\s+([0-9-:\s\.]+)\s+.*?SubmissionStatus\.(?:COMPLETE|PENDING)', content)
                    for m in matches:
                        print(f"Step {idx}: Ref {m[0]} Date {m[1]}")
                except:
                    pass

