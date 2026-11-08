# Name That LLM 0.606 Reproduction Source

This is a code-only source release by **cosmorin**. It is not a checkpoint bundle or a ready-made prediction file.

- Own saved reproduction: script version **360410554**, observed Public **0.606**.
- Public upstream: Joseph Adamski, [Teach the Model + 10% ATLAS V1](https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas?scriptVersionId=360203672), author-reported **0.607**. Later V4 is a different saved version.
- No own independent Public improvement is established by this release. The I/O staging edits were intended to preserve inference calculations; complete numerical parity and original input-byte equality were not established.

## Current required checkpoint is unavailable

The observed public V1 Input page has one `[Private Datasource]`. The recorded historical position matches the LoRA24 Fivefold input. Its named ordinary page currently says “We can't find that page”; that message alone does not distinguish private, removed or another availability issue.

The code's normal `lora24_public` branch is used with nonzero weights and asserts that a `**/lora24/final/COMPLETE.json` marker exists. The successful saved reproduction actually selected the `lightningv08/ntl-lora24-fivefold-training/lora24/final` path. The exact current private source ID, whether it is the only supplier, and a byte-identical accessible public duplicate have not been established. **This source release does not guarantee a fresh full run.** It does not grant access to, contain, rehost or replace another author's private checkpoint.

## Use on your own Kaggle account

1. Join [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) and personally accept its rules. Reading the code does not run a GPU or submit to the competition.
2. Open the pinned upstream V1 or import the supplied source-only Notebook into your own private Notebook. Inspect every input in `dependencies.json`. Do not start a full run while a required input is unavailable; do not treat a removed branch as the same model.
3. If all required permitted inputs are available, select **GPU T4 x2** and **Internet Off**. Preserve input versions; a newer dataset/model is a separate reproduction condition.
4. Choose **Save Version → Save & Run All**. After successful completion, select **Submit to Competition** and `submission.parquet` from the saved output. The code writes columns `id` and `model`, one row per test id. The visible three-row test is a preview; Kaggle regenerates predictions during hidden scoring.

Author scores and the saved reproduction score are not guaranteed for your run. CPU-only execution or a GPU runtime estimate is not promised. No new run was performed to publish this source.

## Files and provenance

- `name_that_llm_0606_source_only.ipynb`: four original saved reproduction code cells retained exactly; all 51 saved outputs removed, all execution counts null, metadata restricted to generic Python information. Introductory documentation was updated; upstream historical credits are retained.
- `dependencies.json`: 24 logical input references with recorded versions and explicit unresolved bindings. It contains links, not data or checkpoint bytes.
- `LICENSE`, `NOTICE`: Apache-2.0 code license and attribution/change notice. The own byte-verification scratch-cache helper is included in the Notebook code, with this release's same code license. The original helper had no separate pre-existing license header.
- `MANIFEST.json`, `checksums.sha256`: contents and checksums of this code-only release.

No original passages, official train/test files, hidden labels, OOF arrays, prediction payloads, personal information, account metadata, internal receipts, screenshots or model weights are included. Third-party data and model terms remain separate; retain their attribution and licenses if obtaining them from their own source.
