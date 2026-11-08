import urllib.request
import json
import os

ENV_PATH = r"C:\Users\s3cr3t\.gemini\antigravity\scratch\kaggle_claude_qa\.env"
env = {}
with open(ENV_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

prompt = """
You are Claude Sonnet (from Kaggle Proxy), serving as the Senior Consultant & Lead Architect on the Deliberative Council for the Kaggle 'Name That LLM' competition (18 classes, metric Balanced Accuracy).

The user has just addressed you directly with urgency, frustration, and weariness:
"claude ơi, bạn hãy phản biện lại coi thử, tôi ở các môc này lâu lắm r"
(Claude, please critique and counter-analyze our situation; I have been stuck at these milestones for way too long!)

Here is the exact reality of the project right now:
1. Leaderboard & CV History:
   - Deep Learning Baseline (Qwen2.5-7B + DeBERTa-v3): 0.631 CV, ~0.663 Public LB.
   - Project has spent weeks hovering around 0.663 - 0.664 (V80 - V92).
   - Champion V92: 0.664 (all-time peak on Public LB).
   - Local 5-fold GroupKFold CV recently achieved 72.73% (+9.72% gain) using Dual-SVC + Sibling Micro-Gates + Sibling Residuals.
   - Champion V93: We tried enabling bipartite Hungarian matching on test data, but V93 suffered a catastrophic collapse to 0.528!
   - Meanwhile, World #1 on Kaggle (Le Viet) is at 0.751, and World Top 10 is at 0.724. The user's goal is >= 75.50%.

2. Root Causes Uncovered:
   - Why V92 was stuck at 0.664 despite local CV being 72.73%:
     In V92, `infer.py` was executed with `--perpassage 1`. Line 905 states:
     `pred = tot.argmax(1) if a.perpassage else hungarian(tot, Gte)`
     Because `perpassage=1`, V92 completely bypassed Hungarian matching on test data and evaluated every passage with pure independent argmax!
     When we simulate test evaluation on local training data using independent argmax, the score is only 49.38% (which maps to 0.664 on LB). With oracle prompt Hungarian, it jumps to 72.73%!
   - Why V93 collapsed to 0.528 when `--perpassage 0` was turned on:
     Setting `perpassage=0` in legacy code triggered two destructive bugs:
     a) It executed `center_groups(Ete, Gte)` on Qwen/DeBERTa embeddings using unsupervised agglomerative clustering with an overly aggressive threshold `0.02`, which clumped hundreds of distinct prompts into massive 100+ sample clusters, corrupting the embeddings.
     b) Fatal Sibling Truncation bug in `infer.py` line 866:
        `_cand_classes = [L2I[s] for s in _siblings[:len(_sib_indices)]]`
        When a cluster only had 2 sisters present, it sliced `_siblings[:2]`, which always mapped to `['claude-opus-4-6', 'claude-opus-4-8']` and NEVER assigned `claude-opus-5`! It systematically erased `claude-opus-5`, `gemini-3.8-flash`, and `gpt-5.6-terra` across the entire test set!

3. Your Mandate as Senior Consultant & Grandmaster Architect:
Provide an incisive, deep, mathematically grounded peer-review critique:
I. The Brutal Truth: Expose the optical illusion of local CV (72.73% with Oracle Prompt Hungarian) vs unlabelled shuffled test passages. Why has the team been running in circles around 0.663?
II. Anatomy of the Failure: Deconstruct V92 (0.664) and V93 (0.528). What went wrong mechanically and conceptually?
III. The Bipartite Prompt Clustering Dilemma: In test time, prompt_id is NOT provided. How do top teams (Le Viet, Meni Gottesman at 0.72 - 0.75) reconstruct prompt groups reliably? Why did agglomerative clustering at threshold 0.02 fail, and how should prompt clustering be engineered cleanly without embedding corruption?
IV. The Sibling Family Bottleneck: Why Deep Learning fails on sister models (Opus 4.6/4.8/5, Gemini 3.7/3.8, GPT Sol/Terra), and why Classical ML must intervene at the right stage (pre-Hungarian vs post-Hungarian).
V. The Decisive Strategic Blueprint for Champion V94: A clear, prioritized, step-by-step roadmap to break the 0.664 ceiling and surge into the 0.70 - 0.73+ leaderboard zone.

Please deliver your critique with empathy for the user's fatigue, but with unflinching scientific rigor, clarity, and actionable engineering wisdom!
"""

data = json.dumps({
    "model": "anthropic/claude-sonnet-5@default",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 8192
}).encode("utf-8")

req = urllib.request.Request(
    "https://mp-staging.kaggle.net/models/openapi/chat/completions",
    headers={"Authorization": "Bearer " + env["MODEL_PROXY_API_KEY"], "Content-Type": "application/json"},
    data=data,
    method="POST"
)

print("Sending comprehensive critique prompt to Claude Sonnet...")
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        msg = res["choices"][0].get("message")
        if msg and msg.get("content"):
            content = msg["content"]
            with open(r"C:\Users\s3cr3t\.gemini\antigravity\scratch\claude_full_critique.txt", "w", encoding="utf-8") as f:
                f.write(content)
            print("SUCCESS! Wrote", len(content), "characters to scratch/claude_full_critique.txt")
        else:
            print("FAILED! Choice 0:", res["choices"][0])
except Exception as e:
    import traceback
    traceback.print_exc()
