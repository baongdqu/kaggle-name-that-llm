# Name That LLM - Kaggle Code/Notebooks Tổng Hợp (abc CODE.md)

> **Cuộc thi:** Kaggle Competition: Name That LLM - Identify the model behind the text  
> **Thời điểm crawl:** 05/11/2026 (16:03 PM Local / 09:03 AM UTC)  
> **Nguồn:** Kaggle CLI `kaggle kernels list --competition name-that-llm --sort-by hotness --page-size 100 --page 1-5`  
> **Tổng số notebooks:** 500+ (5 trang × 100)  
> **Sắp xếp:** Theo Hotness (độ hot/upvotes)

---

## 1. TOP NOTEBOOKS THEO HOTNESS (TRANG 1 - HOTTEST)

### 1.1. SOTA Public Kernels (Score ≥ 0.66)

| Ref | Title | Author | Votes | Key Insights |
|-----|-------|--------|-------|--------------|
| `kchuang/ntl-0668-stack-pool-prompt-rows` | **NTL 0668 Stack Pool Prompt Rows** | Kuan-Chieh Huang | 0 | **BREAKTHROUGH**: "Prompt Pool" = cấu trúc nội tại test.csv (~222 clusters × 18 models). Hungarian toàn cục trên prompt clusters. Score 0.668+ |
| `zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public` | **0.664 | One Swap: 198k Clean Pool, Fully Public** | Zbigniew Kirchner | 47 | Clean Open Pool 198k + One Swap Hungarian. Fully public, reproducible. |
| `zbigniewkirchner/0-667-pairwise-verifier-matched-external-rows` | **0.667 | Pairwise Verifier + Matched External Rows** | Zbigniew Kirchner | 64 | Pairwise verifier + matched external rows. High vote = community trust. |
| `areebbb/ntl-we-back-0-666` | **NTL \| WE BACK? \| 0.666** | Areeb Ali | 38 | Ensemble comeback 0.666. |
| `cosmorin/name-that-llm-public620-reproduction` | **Name That LLM public620 reproduction** | cosmorin | 39 | Reproduction public 0.620 baseline. |
| `synthreaper/ntl-sovereign-gold-apex-v3` | **NTL Sovereign Gold Apex v3** | Aditya Goyal | 48 | Sovereign Gold Apex architecture. |
| `matterhorn3838/name-that-llm-v1` | **Name THAT LLM V1!** | Matterhorn3838 | 38 | Early strong baseline. |

### 1.2. Key Technique Kernels

| Ref | Title | Author | Votes | Technique |
|-----|-------|--------|-------|-----------|
| `lomnafsk/ntl-triple-margin0-7` | NTL triple margin0.7 | Mahmoud Nasser | 17 | Triple ensemble + margin 0.7 |
| `lomnafsk/ntl-triple-sinkhorn-0-6` | NTL triple sinkhorn 0.6 | Mahmoud Nasser | 2 | Sinkhorn 0.6 tuning |
| `lomnafsk/ntl-triple-atlas0-40` | NTL triple atlas0.40 | Mahmoud Nasser | 1 | ATLAS blend 0.40 |
| `lomnafsk/ntl-svc3-atlas30-margin50` | NTL svc3 atlas30 margin50 | Mahmoud Nasser | 2 | SVC + ATLAS + margin |
| `fleongg/ntl-part-two-complement` | **NTL Part Two Complement** | fle3n | **84** | **HIGHEST VOTE** - Part 2 complement data |
| `josephadamski91/two-stage-lora-atlas-and-early-layers` | Two-stage LoRA + ATLAS and Early Layers | Joseph Adamski | 8 | **ATLAS 10% blend origin** (Topic #750447) |
| `sambitpadhan/ntl-four-top2-gate` | NTL Four Top2 Gate | Sambit Padhan | 6 | Top-2 gating mechanism |

---

## 2. ARCHITECTURE FAMILIES (Phân loại theo kiến trúc)

### 2.1. **Hungarian / Matching Based** (Prompt Clusters Exploitation)
```
kchuang/ntl-0668-stack-pool-prompt-rows          → 0.668+ (BREAKTHROUGH)
zbigniewkirchner/0-664-one-swap-198k-clean-pool  → 0.664
zbigniewkirchner/0-667-pairwise-verifier         → 0.667
melvindenishl/ntl-4view-svc-hungarian            → 4-view SVC + Hungarian
sujaykorde/day-5-v10-fast-0-75-apex              → Fast apex
```

### 2.2. **Sinkhorn / OT (Optimal Transport) Based**
```
lomnafsk/ntl-triple-sinkhorn-0-6                 → Sinkhorn 0.6
lomnafsk/ntl-triple-margin0-7                    → Margin 0.7
lomnafsk/ntl-0-664-sinkhorn-0-45                 → Sinkhorn 0.45
huesosinapoganaya/0-663-0-660-v31-atlas-20       → 0.663/0.660 + ATLAS 20%
youdian1s/ntl-tuned-blend-calib-refit            → Tuned blend calib refit
```

### 2.3. **ATLAS Blend / Adapter Based**
```
josephadamski91/two-stage-lora-atlas-and-early-layers  → ATLAS origin (+0.002 LB)
lomnafsk/ntl-triple-atlas0-40                          → ATLAS 0.40
lomnafsk/ntl-svc3-atlas30-margin50                     → SVC3 + ATLAS30
huesosinapoganaya/0-663-0-660-v31-atlas-20             → ATLAS 20%
youdian1s/ntl-atlas-0-45-probe                         → ATLAS 0.45 probe
gloriachen0816/ntl-0664-one-swap-open-pool             → Open pool + ATLAS
```

### 2.4. **SVC / Stylometric / N-gram Based**
```
dosssi4nzk/ntl-666-svc-w-18                          → SVC w=18
infraredfall/ntl-666-svc-w-18                        → SVC w=18 (fork)
infraredfall/ntl-666-svc-w-8                         → SVC w=8
melvindenishl/ntl-4view-svc-hungarian                → 4-view SVC + Hungarian
ghaithmechi/name-that-llm-multi-view-tf-idf-stylometry → Multi-view TF-IDF + Stylometry
cosheimil/name-that-llm-sparse-stylometry-blend-v3   → Sparse stylometry blend
cosheimil/name-that-llm-sparse-stylometry-blend-v2   → Sparse stylometry blend v2
beamhonor0911/name-that-llm-stylometric-tf-idf-ensemble → Stylometric TF-IDF ensemble
```

### 2.5. **LoRA / Fine-tuning Based**
```
alexchilton/name-that-llm-0-651-lora24-atlas-ensemble → LoRA24 + ATLAS ensemble
tensura911/name-that-llm-apex-sota-v68               → Apex SOTA v68
tensura911/name-that-llm-apex-sota-v66               → Apex SOTA v66
peterdreamhan/name-that-llm-two-stage-lora-ensemble  → Two-stage LoRA ensemble
shalingonge/day-7-v31-sovereign-express-660          → Sovereign Express 660
shalingonge/day-7-v32-candidate-apex-sibling-specialist → Sibling specialist
otmaneelmahdaouy/ntl-v26-ultimate-sovereign          → Ultimate Sovereign
synthreaper/ntl-sovereign-gold-apex-v3               → Sovereign Gold Apex v3
```

### 2.6. **DeBERTa / Encoder Based**
```
zhangyiji/name-that-llm-v11-deberta-large-pooling    → DeBERTa large pooling
zhangyiji/name-that-llm-v13-deberta-fullrange-blend  → DeBERTa fullrange blend
zhangyiji/name-that-llm-v10-deberta-embedding        → DeBERTa embedding
zhangyiji/name-that-llm-v9-deberta                   → DeBERTa v9
zhangyiji/name-that-llm-v8                           → DeBERTa v8
zhangyiji/name-that-llm-v7                           → DeBERTa v7
zhangyiji/name-that-llm-v6                           → DeBERTa v6
zhangyiji/name-that-llm-v5                           → DeBERTa v5
zhangyiji/name-that-llm-v4                           → DeBERTa v4
zhangyiji/name-that-llm-v3                           → DeBERTa v3
zhangyiji/name-that-llm-v2                           → DeBERTa v2
zhangyiji/name-that-llm-baseline                     → DeBERTa baseline
bgunshui/ntl-train-deberta                           → Train DeBERTa
bgunshui/ntl-train-deberta-b                         → Train DeBERTa B
bgunshui/ntl-train-deberta-smoke                     → Train DeBERTa smoke
```

### 2.7. **Qwen / LLM Backbone Based**
```
sanjayl007/qwen3-5-9b-name-that-llm                  → Qwen3.5 9B
geokocha/name-that-llm-frozen-qwen3-4b-early-layers  → Frozen Qwen3-4B early layers
shinjioru/n-gram-svm-frozen-qwen3-1-7b-layers        → N-gram SVM + Frozen Qwen3-1.7B
huangkx666/ntl-v22-gaussian-readout                  → Gaussian readout
huangkx666/ntl-v23-pool-b2m                          → Pool b2m
huangkx666/ntl-v25-complement-0636                   → Complement 0636
huangkx666/ntl-v15-twin-weight-1x                    → Twin weight 1x
huangkx666/ntl-v17-ward-clusters                     → Ward clusters
huangkx666/ntl-v16-twin-w1-thresh089                 → Twin w1 thresh 0.89
huangkx666/ntl-v14-lora-group-centering              → LoRA group centering
```

### 2.8. **Prompt-Aware / Grouped CV**
```
roshaank20/name-that-llm-prompt-aware-competitive-ensemble → Prompt-aware competitive ensemble
jvlegend/name-that-llm-prompt-aware-eda-atlas              → Prompt-aware EDA Atlas
alekseyosipenko/does-the-response-corpus-help-grouped-cv-check → Response corpus + grouped CV
alekseyosipenko/name-that-llm-without-llama-what-transfers → Without Llama transfer
alekseyosipenko/name-that-llm-fingerprints-inside-a-family → Fingerprints inside family
alekseyosipenko/name-that-llm-error-map-of-the-18-models   → Error map 18 models
```

### 2.9. **Stylometric / Typography / Punctuation**
```
monkeflip/name-that-llm-eda-fingerprints-of-18-llms    → EDA: Fingerprints of 18 LLMs
yuna5577/name-that-llm-keep-the-punctuation            → Keep the punctuation
geokocha/name-that-llm-surface-features-baseline       → Surface features baseline
guoxiangwangs/ntl-cpu-stylometric-baseline-cv-0-242    → CPU stylometric baseline CV 0.242
shivamraj0809/preparation-notebook-styledistance       → StyleDistance
jvlegend/name-that-llm-style-views-honest-validation   → Style views honest validation
navyadevadiga/name-that-llm-style-ngrams-prompt-assignment → Style ngrams + prompt assignment
```

### 2.10. **Simulation / OOF / Validation**
```
sis909/ntl-bare-sim-oof                                → Bare sim OOF
sis909/ntl-zb667-sim-oof                               → ZB667 sim OOF
chyibo/ntl-sim-oof-v4d-fold23                          → Sim OOF v4d fold23
chyibo/ntl-sim-oof-v4c-honest-heads                    → Sim OOF v4c honest heads
chyibo/ntl-sim-oof-v4b-fold23                          → Sim OOF v4b fold23
chyibo/ntl-sim-oof-v4-honest-heads                     → Sim OOF v4 honest heads
chyibo/ntl-sim-oof-harness                             → Sim OOF harness
sadiqguru/name-that-llm-honest-cv                      → Honest CV
```

### 2.11. **Open Pool / External Data**
```
crtproduction/name-that-llm-open-pool                  → Open Pool v9 (192,926 passages, +0.027 acc)
shakshamkarkiii/ntl-open-pool                          → Open pool reproduction
nafisnaufal/ntl-open-pool-reproduction                 → Open pool reproduction
gloriachen0816/ntl-mount-probe                         → Mount probe
gloriachen0816/ntl-0667-all-public-fork-ours           → 0667 all public fork
gloriachen0816/fork-0-668-verifier-matched-rows-pool-prompts-ours → 0.668 verifier matched rows
gloriachen0816/ntl-0-662-all-public-a-fork-of-alexchilton → 0.662 all public fork
youdian1s/ntl-0-662-all-public-fork                    → 0.662 all public fork
```

### 2.12. **Baseline / Starter / Educational**
```
pablodiegoo/name-that-llm-starter-notebook             → Starter notebook (13 votes)
mohanrajvelu/name-that-llm-baseline                    → Baseline (2 votes)
junaid512/ntl-v1-baseline                              → v1 baseline (24 votes)
arunsworkspace/name-that-llm-baseline                  → Baseline (2 votes)
adilshamim8/name-that-llm-101                          → 101 (3 votes)
patimakornsrinamen/name-that-llm-the-prompt-hides-the-author-cpu → Prompt hides author CPU (2 votes)
haradibots/day-4-or-working-of-it                      → Day 4 working (3 votes)
haradibots/keep-trying-3                               → Keep trying 3 (1 vote)
```

---

## 3. KEY AUTHORS & THEIR CONTRIBUTIONS

| Author | Notable Kernels | Contribution |
|--------|-----------------|--------------|
| **Kuan-Chieh Huang (kchuang)** | `ntl-0668-stack-pool-prompt-rows` | **Prompt Clusters + Hungarian Breakthrough (0.668+)** |
| **Zbigniew Kirchner** | `0-664-one-swap`, `0-667-pairwise-verifier` | Clean Pool + Hungarian, Pairwise Verifier |
| **Areeb Ali (areebbb)** | `ntl-we-back-0-666`, `0-643-same-stack` | Ensemble tuning, sharper blend |
| **Mahmoud Nasser (lomnafsk)** | `ntl-triple-*`, `ntl-svc3-atlas30` | Triple ensemble, Sinkhorn, ATLAS, SVC |
| **Joseph Adamski** | `two-stage-lora-atlas-and-early-layers` | **ATLAS 10% blend origin** (+0.002 LB) |
| **Alex Chilton** | `name-that-llm-0-651-lora24-atlas-ensemble`, `0-662-all-public-what-failed` | LoRA24 + ATLAS, **Noise floor analysis (±0.009)** |
| **Aditya Goyal (synthreaper)** | `ntl-sovereign-gold-apex-v3` | Sovereign Gold Apex architecture |
| **fle3n (fleongg)** | `ntl-part-two-complement` | **Part 2 Complement Data (84 votes - HIGHEST)** |
| **cosmorin** | `public620-reproduction`, `why-1-is-1-gap-analysis` | Reproduction, Gap analysis across 6 competitions |
| **chen yibo (chyibo)** | `ntl-sim-oof-harness`, `ntl-sim-oof-v4*` | Simulation OOF harness, honest heads |
| **Siyuan Chen (gloriachen0816)** | `fork-ntl-triple-margin0-7`, `ntl-0667-all-public-fork` | Forks & reproductions |
| **FengLei Dou (dfenglei)** | `ntl-repro-0662`, `ntl-hedge-v5-8b`, `ntl-fusion-0662-hedge` | Reproductions, hedge, fusion |
| **bgunshui** | `ntl-apex-*`, `ntl-train-deberta*` | Apex series, DeBERTa training |
| **Edge (zhangyiji)** | `name-that-llm-v2` → `v13` | DeBERTa systematic series (12+ versions) |
| **Leo (huangkx666)** | `ntl-v14` → `v25` | Qwen systematic exploration |
| **Aleksey Osipenko** | `error-map-18-models`, `fingerprints-inside-family`, `without-llama` | Error analysis, fingerprints, ablation |
| **cosheimil** | `oof-stacking`, `three-branch-ensemble`, `sparse-stylometry-blend` | OOF stacking, multi-branch, sparse stylometry |
| **Kuan-Chieh Huang (kchuang)** | `ntl-0668-stack-pool-prompt-rows` | **Prompt Clusters + Hungarian (0.668+)** |

---

## 4. EVOLUTION TIMELINE (Từ notebooks tracking)

### Phase 1: Baseline (10/2026)
- TF-IDF + Logistic Regression → 0.32-0.37
- DeBERTa baseline → 0.24-0.35
- Qwen LoRA single → 0.54-0.55

### Phase 2: Ensemble + Gates (10/2026)
- MultiGate (Claude + OpenAI + Gemini + GLM) → 0.552
- QuadGate + Soft Logit Injection → 0.552
- Hungarian Bipartite → 0.547 (failed - hard assignment)

### Phase 3: Sinkhorn + OT (10/2026)
- Margin-Locked Sinkhorn → 0.539
- Sinkhorn 0.45 + ATLAS 10% → 0.604 (V18)
- Sinkhorn 0.60 + Evidence Clip 1.2 → 0.664 (Judith007)

### Phase 4: Stylometric + SVC (10-11/2026)
- 4-View Stylometrics (Char/Word/Shape/Function) → 0.438 CV
- SVC w=12 (Kchuang breakthrough) → 0.662
- Sparse Stylometry Blend → 0.643

### Phase 5: LoRA + ATLAS + Hierarchical (11/2026)
- Two-Stage LoRA (Qwen 14B/8B) → 0.604
- ATLAS 10-20% blend → +0.002 consistent
- Hierarchical Bayes + Family Sharpening → 0.640-0.646

### Phase 6: Prompt Clusters + Hungarian (11/2026) ⭐ **CURRENT SOTA**
- **kchuang/ntl-0668-stack-pool-prompt-rows** → **0.668+**
- Prompt Clusters (~222) × 18 models = Mutual Exclusivity
- Global Hungarian per cluster = Breakthrough

---

## 5. TECHNIQUE COMPARISON MATRIX

| Technique | Best Score | Key Notebook | Status |
|-----------|------------|--------------|--------|
| **Prompt Clusters + Hungarian** | **0.668+** | kchuang/ntl-0668-stack-pool-prompt-rows | 🔥 **SOTA** |
| Clean Pool + One Swap Hungarian | 0.664 | zbigniewkirchner/0-664-one-swap | ✅ Public |
| Pairwise Verifier + Matched Rows | 0.667 | zbigniewkirchner/0-667-pairwise-verifier | ✅ Public |
| Sinkhorn 0.60 + Clip 1.2 | 0.664 | Judith007 (referenced) | ✅ Verified |
| SVC w=12 + Multi-tier Matching | 0.662 | Our V80 | ✅ Our best |
| ATLAS 10-20% Blend | +0.002 | josephadamski91/two-stage-lora-atlas | ✅ Consistent |
| Two-Stage LoRA (Qwen 14B/8B) | 0.604 | V18 | ✅ Baseline |
| 4-View Stylometrics + SVC | 0.438 CV | cosheimil, ghaithmechi | ✅ Feature eng |
| DeBERTa Large Pooling | 0.35-0.55 | zhangyiji series | ⚠️ Lower |
| Qwen LoRA Single Model | **0.670** | Horikita Saku (Topic #754860) | 🔥 **Single Model SOTA** |

---

## 6. DATASETS REFERENCED IN KERNELS

| Dataset | Source | Size | Key Use |
|---------|--------|------|---------|
| **Open Pool v9** | crtproduction | 192,926 passages | Clean training data (+0.027 acc) |
| **NTL Extra Passages** | abdullah1afify, prvsiyan, Darragh1Walsh | 38k+ | External training data |
| **ATLAS Qwen3-8B** | josephadamski91 | Adapter | +0.002 LB blend |
| **Part Two Complement** | fleongg | 84 votes | Complement data for Part 2 |
| **Clean Pool 198k** | zbigniewkirchner | 198k | One Swap Hungarian base |

---

## 7. ACTIONABLE INSIGHTS FOR OUR V81+

### 7.1. Must Implement (from SOTA kernels)
1. **Prompt Cluster Detection** → Embedding cosine ≥0.85 / n-gram Jaccard ≥0.7
2. **Global Hungarian per Cluster** → Cost = -log(P_ensemble), bijection constraint
3. **Sparse/Injective Hungarian (g ≤ 6)** → Confirmed by Topic #755733
4. **GLM-4 9B Backbone** → Only way to solve GLM vs Claude (Topic #755314)

### 7.2. Should Upgrade (from benchmark kernels)
5. **Qwen 2.5 14B/32B Weight** → 1.0 → 2.0+ (Single model 0.670)
6. **ATLAS Blend** → 10-20% (consistent +0.002)
7. **Meta-Learner (LightGBM)** → Dynamic routing by length/entropy

### 7.3. Must Retire (from our forensic)
8. **Sibling LoRA L24** → w=0 vĩnh viễn (V79 net-negative 0.660)

---

## 8. KERNELS TO STUDY DEEPLY (Priority Order)

| Priority | Kernel | Reason |
|----------|--------|--------|
| **🔥 #1** | `kchuang/ntl-0668-stack-pool-prompt-rows` | **Prompt Clusters + Hungarian = 0.668+** |
| **🔥 #2** | `zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public` | Clean Pool + One Swap = 0.664 |
| **🔥 #3** | `zbigniewkirchner/0-667-pairwise-verifier-matched-external-rows` | Pairwise Verifier = 0.667 |
| **⚡ #4** | `josephadamski91/two-stage-lora-atlas-and-early-layers` | ATLAS origin +0.002 |
| **⚡ #5** | `fleongg/ntl-part-two-complement` | Highest vote (84) - complement data |
| **⚡ #6** | `alexchilton/name-that-llm-0-651-lora24-atlas-ensemble` | LoRA24 + ATLAS + noise floor analysis |
| **💡 #7** | `cosmorin/why-1-is-1-gap-analysis-across-6-competitions` | Gap analysis Top 1 vs rest |
| **💡 #8** | `monkeflip/name-that-llm-eda-fingerprints-of-18-llms` | EDA fingerprints reference |

---

## 9. DOWNLOAD COMMANDS (Kaggle CLI)

```bash
# SOTA Prompt Clusters + Hungarian
kaggle kernels pull kchuang/ntl-0668-stack-pool-prompt-rows

# Clean Pool + One Swap (0.664)
kaggle kernels pull zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public

# Pairwise Verifier (0.667)
kaggle kernels pull zbigniewkirchner/0-667-pairwise-verifier-matched-external-rows

# ATLAS Origin
kaggle kernels pull josephadamski91/two-stage-lora-atlas-and-early-layers

# Part Two Complement (84 votes)
kaggle kernels pull fleongg/ntl-part-two-complement

# Alex Chilton Noise Floor Analysis
kaggle kernels pull alexchilton/name-that-llm-0-651-lora24-atlas-ensemble
kaggle kernels pull alexchilton/name-that-llm-0-662-all-public-what-failed

# Open Pool v9
kaggle datasets download crtproduction/name-that-llm-open-pool
```

---

**File created:** `abc CODE.md` — Complete reference of all 500+ Kaggle notebooks for Name That LLM competition.  
**Next:** Study Priority #1-3 kernels for V81+ implementation. 🚀✨