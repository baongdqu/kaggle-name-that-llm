import numpy as np
from scipy.optimize import linear_sum_assignment

def solve_injective_hungarian(log_probs):
    """
    Given log_probs matrix of shape (n_passages, 18), where n_passages <= 18.
    Finds the injective mapping maximizing total log probability.
    Cost = -log_probs.
    """
    n, k = log_probs.shape
    cost = -log_probs
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind

# Test with 3 passages and 18 classes
np.random.seed(42)
test_lp = np.log(np.random.dirichlet(np.ones(18), size=3))
# Make passage 0 and 1 both favor class 2
test_lp[0, 2] = 5.0
test_lp[1, 2] = 4.5
test_lp[1, 5] = 4.0
test_lp[2, 7] = 6.0

best_classes = solve_injective_hungarian(test_lp)
print("Argmax per passage (could have duplicate):", test_lp.argmax(axis=1))
print("Hungarian injective assignment (guaranteed unique):", best_classes)
assert len(set(best_classes)) == len(best_classes), "Must be distinct!"
print("Hungarian test passed successfully!")
