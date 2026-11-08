import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# 30 versions data
versions = [f"V{i}" for i in range(1, 31)]
v_nums = list(range(1, 31))

scores = [
    0.550, 0.551, 0.547, 0.552, 0.549, 0.539, 0.539, 0.541, 0.548, 0.544,
    0.552, 0.552, 0.557, 0.577, 0.599, 0.603, 0.602, 0.604, 0.600, 0.639,
    0.612, 0.639, 0.627, 0.639, 0.630, 0.635, 0.639, 0.632, 0.638, 0.646
]

dates = [
    "16/10", "16/10", "17/10", "17/10", "17/10", "17/10", "18/10", "18/10", "18/10", "18/10",
    "18/10", "19/10", "19/10", "19/10", "19/10", "19/10", "24/10", "24/10", "24/10", "24/10",
    "25/10", "25/10", "25/10", "25/10", "25/10", "26/10", "26/10", "26/10", "26/10", "26/10"
]

# Calculate running peak
running_peak = []
current_max = 0
for s in scores:
    if s > current_max:
        current_max = s
    running_peak.append(current_max)

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={'height_ratios': [2.2, 1.2]}, dpi=150)
fig.patch.set_facecolor('#0d1117')
for ax in (ax1, ax2):
    ax.set_facecolor('#161b22')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    ax.tick_params(colors='#8b949e', which='both')
    ax.grid(True, linestyle='--', alpha=0.25, color='#8b949e')

# --- SUBPLOT 1: Full 30 versions progression ---
ax1.plot(v_nums, scores, marker='o', markersize=6, color='#58a6ff', linewidth=1.8, 
         alpha=0.75, label='Điểm từng phiên bản (Submission Score)')
ax1.plot(v_nums, running_peak, marker='s', markersize=7, color='#f0883e', linewidth=3.0, 
         label='Kỷ lục lũy kế toàn đội (Running Team Record)')
ax1.fill_between(v_nums, running_peak, 0.535, color='#f0883e', alpha=0.08)

# Baseline line
ax1.axhline(0.547, color='#8b949e', linestyle=':', linewidth=1.5, label='Mức sàn khởi điểm (Baseline: 0.547)')

# Day boundary dividers
day_dividers = [2.5, 6.5, 11.5, 16.5, 20.5, 25.5]
for d in day_dividers:
    ax1.axvline(d, color='#ff7b72', linestyle='--', alpha=0.3, linewidth=1.2)

# Annotations for major breakthroughs
key_annotations = [
    (2, 0.551, "V2: 0.551\n(Top 8)", (15, 12)),
    (4, 0.552, "V4: 0.552\n(QuadGate)", (-20, 18)),
    (13, 0.557, "V13: 0.557\n(Phá trần)", (-30, 15)),
    (14, 0.577, "V14: 0.577\n(Rank 33)", (-40, 15)),
    (15, 0.599, "V15: 0.599\n(Top 13 HCV)", (-50, 15)),
    (16, 0.603, "V16: 0.603\n(Mốc 0.600+)", (10, 12)),
    (20, 0.639, "V20: 0.639\n(+0.035 - Rank 135)", (-60, 18)),
    (24, 0.639, "V24: 0.639\n(Golden Pool)", (10, -28)),
    (30, 0.646, "V30: 0.646 *\n(Rank 128 Toan cau)", (-55, 18)),
]

for v_idx, score, text, offset in key_annotations:
    ax1.annotate(
        text,
        xy=(v_idx, score),
        xytext=offset,
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        color='#ffffff',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#58a6ff', alpha=0.9),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='#58a6ff', lw=1.2)
    )

ax1.set_title("TIEN TRINH GIA TANG DIEM SO KAGGLE PUBLIC LB (TU V1 DEN V30)", 
              fontsize=16, fontweight='bold', color='#f0f6fc', pad=15)
ax1.set_ylabel("Public Leaderboard Score", fontsize=12, fontweight='bold', color='#c9d1d9')
ax1.set_ylim(0.535, 0.655)
ax1.set_xlim(0.5, 30.5)
ax1.set_xticks(v_nums)
ax1.set_xticklabels(versions, rotation=0, fontsize=9, fontweight='bold')
ax1.legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d', fontsize=10)

# --- SUBPLOT 2: Daily peak progression ---
daily_dates = ["16/10", "17/10", "18/10", "19/10", "24/10", "25/10", "26/10"]
daily_peaks = [0.551, 0.552, 0.552, 0.603, 0.639, 0.639, 0.646]
x_days = list(range(len(daily_dates)))

ax2.plot(x_days, daily_peaks, marker='D', markersize=9, color='#7ee787', linewidth=2.8, 
         label='Ky luc diem so theo ngay')
ax2.fill_between(x_days, daily_peaks, 0.540, color='#7ee787', alpha=0.12)

# Value labels on subplot 2
deltas = ["+0.004", "+0.001", "0.000", "+0.051 [MAX]", "+0.036 [JUMP]", "0.000", "+0.007 [PEAK]"]
for i, (d, p, delta) in enumerate(zip(daily_dates, daily_peaks, deltas)):
    ax2.annotate(
        f"{p:.3f}\n({delta})",
        xy=(i, p),
        xytext=(0, 12),
        textcoords='offset points',
        ha='center',
        fontsize=9.5,
        fontweight='bold',
        color='#7ee787' if '+' in delta else '#c9d1d9',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#21262d', edgecolor='#30363d', alpha=0.85)
    )

ax2.set_title("DIEM DINH KY LUC THEO TUNG NGAY THI DAU & MUC TANG TRUONG (DELTA)", 
              fontsize=13, fontweight='bold', color='#f0f6fc', pad=12)
ax2.set_xlabel("Ngay Thi Dau", fontsize=11, fontweight='bold', color='#c9d1d9', labelpad=8)
ax2.set_ylabel("Diem Ky Luc", fontsize=11, fontweight='bold', color='#c9d1d9')
ax2.set_ylim(0.540, 0.655)
ax2.set_xlim(-0.5, 6.5)
ax2.set_xticks(x_days)
ax2.set_xticklabels(daily_dates, fontsize=10, fontweight='bold')
ax2.legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d', fontsize=9.5)

plt.tight_layout()

# Save to conversation artifacts
out_path = r"C:\Users\s3cr3t\.gemini\antigravity-ide\brain\f5ca0093-20f3-4308-a03b-7ccba8e6503a\score_progression_chart.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Chart successfully saved to {out_path}!")
