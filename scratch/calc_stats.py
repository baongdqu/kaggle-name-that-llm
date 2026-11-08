import numpy as np

# Kernel runtimes in seconds from logs / verified runs
kernel_runtimes_sec = {
    "V1 (57285498)": 1946,
    "V2 (57285636)": 1950,
    "V3 (57290359)": 1810,
    "V4 (57294743)": 1966,
    "V5 (57294881)": 2104,
    "V6 (57296505)": 2150,
    "V7 (57330867)": 2000,
    "V8 (57330913)": 2050,
    "V9 (57336421)": 1936,
    "V10 (57336428)": 2766,
    "V11 (57359637)": 2236,
    "V12 (57373323)": 1978,
    "V13 (57383708)": 1951,
    "V14 (57388892)": 2327,
    "V15 (57394349)": 2357,
    "V16 (57399922)": 2649,
    "V17 (57580018)": 2591,
}

times_m = [s / 60.0 for s in kernel_runtimes_sec.values()]
print("=== KERNEL RUNTIME STATS (Minutes) ===")
print(f"Count: {len(times_m)}")
print(f"Mean:   {np.mean(times_m):.2f} mins ({np.mean(times_m)*60:.1f} secs)")
print(f"Median: {np.median(times_m):.2f} mins")
print(f"Min:    {np.min(times_m):.2f} mins ({list(kernel_runtimes_sec.keys())[np.argmin(times_m)]})")
print(f"Max:    {np.max(times_m):.2f} mins ({list(kernel_runtimes_sec.keys())[np.argmax(times_m)]})")
print(f"Std:    {np.std(times_m):.2f} mins")

# Estimated active scoring runtimes on private test set (4000 rows)
# In code competitions, private test scoring runs the full test set:
# For V1-V3: ~90-100m
# For V4-V8: ~60-70m
# For V9-V12: ~60-75m
# For V13-V16: ~75-90m
scoring_runtimes_m = {
    "V1": 95,
    "V2": 100,
    "V3": 75,
    "V4": 65,
    "V5": 68,
    "V6": 70,
    "V7": 65,
    "V8": 65,
    "V9": 70,
    "V10": 80,
    "V11": 65,
    "V12": 60,
    "V13": 70,
    "V14": 80,
    "V15": 85,
    "V16": 88,
}
s_times = list(scoring_runtimes_m.values())
print("\n=== SCORING RUNTIME STATS (Minutes) ===")
print(f"Count: {len(s_times)}")
print(f"Mean:   {np.mean(s_times):.2f} mins")
print(f"Median: {np.median(s_times):.2f} mins")
print(f"Min:    {np.min(s_times):.2f} mins ({list(scoring_runtimes_m.keys())[np.argmin(s_times)]})")
print(f"Max:    {np.max(s_times):.2f} mins ({list(scoring_runtimes_m.keys())[np.argmax(s_times)]})")
