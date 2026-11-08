import numpy as np
from scipy.special import logsumexp, softmax

# Simulate an ensemble probability matrix with 10 rows and 18 classes
np.random.seed(42)
logits = np.random.randn(10, 18) * 2.0
P = softmax(logits, axis=1)

LABELS = [
 'claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max'
]

FAMILY_IDX = {f: np.array([i for i, l in enumerate(LABELS) if l.startswith(f)]) for f in ('claude', 'gpt', 'gemini', 'glm')}

# Verify mass conservation
for fam, idx in FAMILY_IDX.items():
    orig_family_mass = P[:, idx].sum(axis=1)
    
    # Simulate gate scores
    gate_scores = np.random.randn(10, len(idx))
    gate_lp = gate_scores - logsumexp(gate_scores, axis=1, keepdims=True)
    
    base = logits[:, idx].copy()
    base_lp = base - logsumexp(base, axis=1, keepdims=True)
    base_p = np.exp(base_lp)
    
    sp = np.sort(base_p, axis=1)
    margin = sp[:, -1] - sp[:, -2]
    confidence_gate = np.clip((0.35 - margin) / 0.35, 0.0, 1.0)
    strength = 0.25 * confidence_gate
    
    mixed_lp = (1.0 - strength[:, None]) * base_lp + strength[:, None] * gate_lp
    mixed_lp = mixed_lp - logsumexp(mixed_lp, axis=1, keepdims=True)
    
    family_mass = logsumexp(base, axis=1)
    logits[:, idx] = family_mass[:, None] + mixed_lp
    
    new_P = softmax(logits, axis=1)
    new_family_mass = new_P[:, idx].sum(axis=1)
    
    diff = np.abs(orig_family_mass - new_family_mass).max()
    print(f"Family {fam}: max mass deviation = {diff:.1e}")
    assert diff < 1e-6, f"Mass not conserved for {fam}!"

print("ALL FAMILY PROBABILITY MASSES ARE 100% CONSERVED!")

