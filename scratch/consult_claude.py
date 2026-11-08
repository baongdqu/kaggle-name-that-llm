import subprocess, sys

prompt = """We are competing in Kaggle "Name That LLM" (18 classes, Balanced Accuracy metric).
Data properties:
- Train: 2,699 samples (150 prompts x 18 responses per prompt).
- Texts are flattened into single lines (no newline characters, no markdown headers, average 115 words).
- 18 classes include several sister-model clusters:
  1. gemini-flash: gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash (0 em-dashes, very similar style)
  2. gpt-5.6 trio: gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra
  3. glm/claude: glm-5.3, glm-5.3-flash, claude-fable-5
  4. claude-opus trio: claude-opus-4-6, claude-opus-4-8, claude-opus-5
  5. other anchors: gpt-5.5, gpt-6-astra, kimi-k3, qwen3.8-max, claude-sonnet-5, gemini-3.1-pro-preview

Current performance:
- Pure Deep Learning OOF ensemble (Qwen3-14B + Qwen3-8B + DeBERTa-v3-large) + Hungarian bipartite matching per prompt = 63.01% Balanced Accuracy.
- Deep Learning + Global LinearSVC (Multi-view TFIDF: char 3-5, word 1-3, shape, function words) + Hungarian = 66.31% Balanced Accuracy.
- Diagnostic analysis shows ~400 errors are concentrated in just 5 sister pairs (gemini 3.7 vs 3.8, gpt-5.6 sol vs terra, glm-5.3 vs glm-5.3-flash, claude-fable vs glm-5.3-flash, opus 4.8 vs 4.6 vs 5).

Our goal is to reach >= 75.5% Balanced Accuracy.
As Senior Strategist & Thinker, analyze:
1. What exact stylometric, forensic, or statistical features can cleanly separate these sister models where Deep Learning embeddings and standard TF-IDF saturate?
2. What is the optimal multi-stage / hierarchical architecture (e.g., Macro-family router -> Micro-cluster pairwise rerankers vs joint cost matrix adjustment)?
3. How to formulate the prompt-level assignment (Hungarian / Sinkhorn / Anchor-locking) to prevent sister-model ambiguity from degrading anchor predictions?
Be precise with mathematical formulation and concrete engineering techniques."""

with open("scratch/claude_query.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

cmd = ["py", "-3", r"C:\Users\s3cr3t\.gemini\antigravity\scratch\kaggle_claude_qa\ask_claude.py", "--max-tokens", "16384", prompt]
res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)
