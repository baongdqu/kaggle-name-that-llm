# Name That LLM: free Kaggle Model Proxy generations, with a settings sweep

Single-paragraph passages from the competition's reachable labels, generated through the **Kaggle Benchmarks Model Proxy on
Kaggle's free AI quota** (no paid key). No competition text, id or prompt was sent to any model or is included here.

## Protocol
- **Prompts:** the CC-BY 4.0 1,000-prompt bank of
  [seyitkaangunes/ntl-extra-api-passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages)
  (`prompt_id` g0000-g0999). We work downward from g0999, while
  [abdullah1afify/ntl-generations](https://www.kaggle.com/datasets/abdullah1afify/ntl-generations) works upward from
  g0000. Every label therefore answers the same prompts, and its passages line up with the 4,998 Claude passages.
- **Request:** one user message, no system prompt: `<prompt> Write a single paragraph of about N words, with no title
  or headings.` Fresh conversation per call, `max_tokens` 16000, no temperature/top_p/seed.
- **Length:** N is drawn per (prompt, label) from that label's word counts in the competition train set (numbers only),
  clipped to [36, 274].
- **Settings:** `setting` = `default` (provider defaults) or the `reasoning_effort` value sent (`none`, `low`,
  `medium`, `high`). Settings unsupported by a model (HTTP 400, e.g. gpt-6-astra with `none`) were skipped.
- **Cleaning:** whitespace runs collapsed to one space. Only `finish_reason == stop`, non-empty answers whose returned
  model id matches the requested slug are kept.

## Fidelity notes (vs the competition train set)
- **Gemini.** Train Gemini passages contain no em dashes and few hyphenated compounds. Proxy Gemini with
  `reasoning_effort=none` uses em dashes in 30-43 % of passages; with default (thinking) settings, 7 %. Prefer the
  thinking settings for Gemini. Filtering out passages that contain an em dash is a further option.
- **GPT.** All settings are close. `low` was recognised best by a train-only classifier in a 30-prompt pilot.

## Columns (`ntl_proxy_generations.csv`)
`label` (competition label), `prompt_id` (joins the seyitkaangunes prompt bank), `setting`, `target_words`,
`n_words`, `text`, `model_seen` (returned proxy model id), `reasoning_tokens`, `cost_usd` (free-quota meter), `batch`.

## Licence
CC BY 4.0 for the texts. Prompts: seyitkaangunes (CC BY 4.0). The upstream providers' output terms behind Kaggle's
proxy are not public, so the licence is a best-effort statement. Published to satisfy competition rule 2.6 (external
data must be publicly available).
