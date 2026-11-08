import numpy as np

fam_sizes = np.array([5, 4, 2, 5, 1, 1], dtype=np.float64)
priors = fam_sizes / 18.0

# Case 1: Uninformative prior (flat probability 1/18 across all 18 classes)
P_uninform = priors.copy()
odds_1 = P_uninform / priors
print("Uninformative odds:", odds_1)
odds_sharp_1 = np.power(odds_1, 1.22)
P_sharp_1 = odds_sharp_1 * priors
P_sharp_1 /= P_sharp_1.sum()
print("P_sharp_1:", P_sharp_1)
print("Difference from uninformative:", np.abs(P_sharp_1 - priors).max())

# Case 2: Realistic sample where Qwen is predicted at 0.20 (normally 0.0556)
P_qwen_favored = priors.copy()
P_qwen_favored[4] = 0.20 # Qwen index is 5 in 0..5, say index 4 or 5
P_qwen_favored /= P_qwen_favored.sum()
print("\nRaw P with Qwen favored:", P_qwen_favored[4])
# Old V33 method:
P_v33 = np.power(P_qwen_favored, 1.22)
P_v33 /= P_v33.sum()
print("V33 sharpened Qwen P:", P_v33[4])

# Prior-calibrated method:
odds_2 = P_qwen_favored / priors
odds_sharp_2 = np.power(odds_2, 1.22)
P_v34 = odds_sharp_2 * priors
P_v34 /= P_v34.sum()
print("V34 Prior-calibrated sharpened Qwen P:", P_v34[4])
