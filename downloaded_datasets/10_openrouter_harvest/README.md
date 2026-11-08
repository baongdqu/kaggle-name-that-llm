# Name That LLM: OpenRouter Harvest

10,772 single-paragraph passages written by 14 of the 18 models in the Kaggle competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) (`name-that-llm`). They were generated in October 2026 through OpenRouter, with every request pinned to the model's own official provider. The generation protocol follows the competition's training passages as closely as we could reproduce it. The data is meant as extra labelled training text for the competition, and it is released publicly so that any team may use it, as the competition rules require for external generated data.

## What is in it

| File | Rows | Contents |
| --- | --- | --- |
| `passages.csv` | 10,772 | The cleaned passages, one per row, with the label, the served model and provider, the reasoning setting, the prompt, target and actual word counts, token usage and cost. This is the file to train on. |
| `calls.jsonl` | 12,308 | Every API call of the main run as returned, including the calls removed by the cleaning below and the 1,000 superseded Kimi `off` calls. Text keeps its original line breaks. |
| `trials.jsonl` | 520 | The setting trials (prompts 20–59) used to choose a reasoning setting per model, including the settings that were rejected. |
| `setting_trials.jsonl` | 1,210 | The second round of setting trials (prompts 60–219) for Kimi, Qwen and Gemini 3.8 Flash: temperature, top_p, reasoning effort, system prompt, instruction wording and the Vertex route. 160 Kimi `high` calls are also in `calls.jsonl`. |
| `pilot.jsonl` | 300 | The first pilot (prompts 0–19, 13 model ids, mostly default reasoning). 240 of these calls are also in `calls.jsonl`. |
| `prompts.csv` | 1,000 | The prompts with domain, task type and topic. |

### `passages.csv` columns

- `id`: the first 12 hex characters of the SHA-1 of `text`.
- `label`: the competition label the model answers to, e.g. `gpt-5.6-sol`.
- `requested_model` / `served_model` / `served_provider`: the OpenRouter id requested, and what OpenRouter reports it served.
- `setting`: the reasoning setting, one of `default` (API defaults), `off` (`reasoning.enabled = false`), `low` (`reasoning.effort = "low"`) or `high` (`reasoning.effort = "high"`).
- `prompt_id`, `prompt`, `domain`, `task_type`, `topic`: the prompt and its tags.
- `target_words`: the length asked for. `n_words`: the passage's whitespace word count.
- `text`: the passage, with whitespace collapsed to single spaces.
- `finish_reason`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `cost` (USD): as reported by OpenRouter.

## How it was generated

- **Prompts:** 1,000 English prompts from [seyitkaangunes/ntl-extra-api-passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages) (CC BY 4.0), written for that dataset in the style of the competition's topics rather than taken from the competition.
- **Request:** one user message, with no system prompt: the prompt followed by `Write a single paragraph of about N words, with no title or headings.` N is drawn at random from the word counts of the competition's training passages. Sampling uses the API defaults, and `max_tokens` is 8,000 so reasoning cannot crowd out the answer.
- **Routing:** each request is pinned to the model's official provider with fallbacks disabled (OpenAI, Google AI Studio, Alibaba, Z.AI, Moonshot AI). No third-party host served an open-weight model.
- **Post-processing:** a bold wrapper around the whole answer (`**...**`) is removed, as the competition host did. Nothing else is rewritten.
- **Reasoning setting:** each model's setting was chosen on 40 trial prompts by fidelity (below). Gemini, Qwen and GLM refuse reasoning off through their official providers.

## Cleaning

Of the 12,308 calls in `calls.jsonl`, 10,772 passages are kept. These were removed:

- Kimi K3 calls with reasoning `off`, superseded by reasoning effort `high` (version 2).

- Calls cut off at the token limit (`finish_reason` other than `stop`).
- Answers with more than one paragraph, or with markdown (bold, headings, lists, code).
- Answers outside the training word range (36–274 words).
- Gemini passages containing an em dash. The competition's Gemini training passages contain none, while harvested Gemini text uses one in about 22% of passages.
- Duplicates, and any text that repeats a competition training or test passage.

## Per label

| Label | Setting | Passages | Served by | Cost (USD) | Fidelity, exact | Fidelity, lab |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-5.5 | default | 1,000 | OpenAI | 14.80 | 0.78 | 1.12 |
| gpt-5.6-luna | default | 1,000 | OpenAI | 0.95 | 0.68 | 0.98 |
| gpt-5.6-sol | default | 1,000 | OpenAI | 13.58 | 0.83 | 0.96 |
| gpt-5.6-terra | default | 1,000 | OpenAI | 7.94 | 0.96 | 1.04 |
| gpt-6-astra | default | 1,000 | OpenAI | 20.02 | 0.87 | 1.00 |
| gemini-3.1-pro-preview | default | 20 | Google AI Studio | 1.23 | — | — |
| gemini-3.6-flash | low | 776 | Google AI Studio | 8.71 | 0.52 | 0.75 |
| gemini-3.7-flash | low (19 default) | 772 | Google AI Studio | 4.76 | 0.65 | 0.70 |
| gemini-3.8-flash | default | 838 | Google AI Studio | 11.13 | 0.47 | 0.82 |
| glm-5.3 | low | 975 | Z.AI | 0.79 | 0.94 | 0.92 |
| glm-5.3-flash | default | 978 | Z.AI | 1.29 | 0.78 | 0.80 |
| kimi-k3 | high | 996 | Moonshot AI | 12.26 | 0.93 | 0.93 |
| qwen3.8-max | default | 417 (357 `-prime`, 60 `-0902`) | Alibaba | 11.33 | 0.55 / 0.36 | 0.55 / 0.36 |

The five Claude labels and gemini-3.1-pro-preview (beyond 20 pilot passages) were not harvested; public sets already cover them well. Costs are those recorded in `calls.jsonl`. Including the pilot and both rounds of setting trials, the project spent about $125.

**Fidelity** measures how much a model's passages read like its competition training passages. A stylegram classifier (character, word and punctuation n-grams with a linear SVM) is trained on the competition's `train.csv` only. Fidelity is its recall on these passages, divided by its out-of-fold recall on the training passages of the same label. "Exact" counts the right label; "lab" counts the right model family. For reference, the public protocol-matched datasets score 0.69–0.82 exact.

## Known issues

- **Kimi K3 needs reasoning effort `high`.** It scores 0.93 exact. Thinking off scored 0.24 and default reasoning 0.14–0.18, and sampling, system-prompt and wording changes to `off` did not help (`setting_trials.jsonl`).
- **Qwen and Gemini 3.8 Flash** also sit below the public protocol sets (0.55 and 0.47 exact). Qwen's `low` effort still reasoned for about 3,000 tokens and scored 0.05, so only default reasoning was kept.
- **Gemini Flash versions:** the AI Studio route served `gemini-3.7-flash` as requested. A train-only classifier labels those passages 3.7 more often than 3.8 (0.27 vs 0.14), but it separates the Flash versions weakly, so this does not prove which snapshot was served.
- **Prompt topics differ from the competition's.** These are new prompts, so a model trained on them should be validated on the competition's own passages, with prompt-grouped folds.

## Versions

- **2** (2026-10-18): Kimi K3 regenerated at reasoning effort `high` (fidelity 0.24 → 0.93), and `setting_trials.jsonl` added.
- **1** (2026-10-18): first release.

## Licence and attribution

CC BY 4.0. The prompts come from [seyitkaangunes/ntl-extra-api-passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages) (CC BY 4.0). The passages are model outputs generated by the dataset author through each provider's API under its terms of use.
