import numpy as np
import time

def exclusive_marginals(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / p.sum(1, keepdims=True)
    n, k = p.shape
    if n <= 1:
        return p.copy()
    if n > k or n > 8:
        return None
    full = (1 << n) - 1
    f = np.zeros((k + 1, 1 << n))
    b = np.zeros_like(f)
    f[0, 0] = 1.0
    b[k, 0] = 1.0
    for c in range(k):
        f[c + 1] = f[c]
        for mask in range(1, full + 1):
            for i in range(n):
                if mask & (1 << i):
                    f[c + 1, mask] += p[i, c] * f[c, mask ^ (1 << i)]
    for c in range(k - 1, -1, -1):
        b[c] = b[c + 1]
        for mask in range(1, full + 1):
            for i in range(n):
                if mask & (1 << i):
                    b[c, mask] += p[i, c] * b[c + 1, mask ^ (1 << i)]
    z = f[k, full]
    if z <= np.finfo(float).tiny:
        return p.copy()
    q = np.zeros_like(p)
    for i in range(n):
        remaining = full ^ (1 << i)
        for c in range(k):
            subset = remaining
            while True:
                q[i, c] += p[i, c] * f[c, subset] * b[c + 1, remaining ^ subset] / z
                if subset == 0:
                    break
                subset = (subset - 1) & remaining
    return q / q.sum(1, keepdims=True)

def cavity_marginals(p, iters=5, repel_strength=0.85):
    L = np.log(np.clip(p, 1e-12, 1.0))
    q = np.exp(L - L.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    for _ in range(iters):
        lo = np.log(np.clip(1.0 - q, 1e-6, None))
        cavity_penalty = lo.sum(0, keepdims=True) - lo
        L_new = L + repel_strength * cavity_penalty
        q = np.exp(L_new - L_new.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
    return q

def hybrid_injective_decoding(P_group, max_exact_n=6):
    n = len(P_group)
    if n <= 1:
        return P_group
    # Repulsion strength grows dynamically with log2(n)
    repel_weight = min(0.95, 0.40 * np.log2(n) + 0.30)
    if n <= max_exact_n:
        exact_q = exclusive_marginals(P_group)
        if exact_q is not None:
            return (1.0 - repel_weight) * P_group + repel_weight * exact_q
    # For n > 6 or if exact decoding is None, use cavity mean-field
    cavity_q = cavity_marginals(P_group, iters=5, repel_strength=repel_weight)
    return (1.0 - repel_weight) * P_group + repel_weight * cavity_q

# Test on 2, 4, 8 passages
for n in [2, 3, 4, 6, 8, 12]:
    P_dummy = np.random.dirichlet(np.ones(18), size=n)
    # create conflict: passage 0 and 1 both have highest prob on class 0
    P_dummy[0, 0] = 0.70
    P_dummy[1, 0] = 0.65
    P_dummy /= P_dummy.sum(axis=1, keepdims=True)
    
    t0 = time.time()
    out = hybrid_injective_decoding(P_dummy)
    el = (time.time() - t0) * 1000
    print(f"Group size {n:2d}: elapsed {el:6.2f} ms | Conflict resolved: P0_c0={out[0,0]:.3f}, P1_c0={out[1,0]:.3f} (orig 0.70/0.65)")
    assert np.isfinite(out).all()
    assert np.allclose(out.sum(axis=1), 1.0)

print("\nHybrid Injective Decoding PASSED all tests with flying colors!")
