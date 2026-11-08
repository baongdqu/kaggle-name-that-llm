"""Collect benchmark task outputs into passages.csv (one row per passage) plus a raw envelope file.

Usage: collect.py <download dir> [<download dir> ...] --bank <prompt bank csv> --out <dir>
"""
import argparse
import glob
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

LABELS = ["claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "gemini-3.1-pro-preview",
          "gemini-3.6-flash", "gemini-3.8-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"]


def label_of(model_id):
    m = model_id.split("/")[-1].split("@")[0]
    m = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", m)
    return m if m in LABELS else None


def clean(text):
    # Same light cleanup the hosts describe: strip whitespace and a bold wrapper around the whole passage.
    t = text.strip()
    if t.startswith("**") and t.endswith("**") and t.count("**") == 2:
        t = t[2:-2].strip()
    return re.sub(r"[ \t]+", " ", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, raw = [], []
    for d in a.dirs:
        for f in sorted(glob.glob(f"{d}/**/gen__*.jsonl", recursive=True)):
            run = Path(f).parent
            meta = {}
            for rj in run.glob("*.run.json"):
                j = json.load(open(rj))
                meta = dict(run_start=j.get("startTime") or j.get("start_time"), task_version=run.parent.parent.name)
            for line in open(f):
                r = json.loads(line)
                raw.append(r)
                if not r["ok"]:
                    continue
                resp = r["response"]
                ch = resp["choices"][0]
                text = ch["message"].get("content") or ""
                u = resp.get("usage") or {}
                cost = u.get("cost") or {}
                det = u.get("completion_tokens_details") or {}
                rows.append(dict(
                    prompt_id=r["prompt_id"], label=label_of(r["model_requested"]), model_requested=r["model_requested"],
                    model_served=resp.get("model"), setting="default", target_words=r["target_words"],
                    text=clean(text), finish_reason=ch.get("finish_reason"), prompt_tokens=u.get("prompt_tokens"),
                    completion_tokens=u.get("completion_tokens"), reasoning_tokens=det.get("reasoning_tokens"),
                    cost_usd=((cost.get("input_tokens_cost_nanodollars") or 0) + (cost.get("output_tokens_cost_nanodollars") or 0)) / 1e9,
                    seconds=r["seconds"], response_id=resp.get("id"), **meta))
    df = pd.DataFrame(rows)
    df["n_words"] = df.text.str.split().str.len()
    df["served_matches"] = df.model_served.map(label_of) == df.label
    df["n_paragraphs"] = df.text.str.count(r"\n\s*\n") + 1
    df["has_markdown"] = df.text.str.contains(r"(^|\n)\s*(#|[-*] |\d+\. )|\*\*", regex=True)
    df["has_em_dash"] = df.text.str.contains("\u2014")
    df["passage_ok"] = (df.finish_reason == "stop") & df.served_matches & (df.n_paragraphs == 1) & ~df.has_markdown \
        & df.n_words.between(20, 320) & df.label.notna()
    df = df.drop_duplicates(["prompt_id", "model_requested", "setting"], keep="last")
    df.insert(0, "passage_id", [hashlib.sha256(f"{p}|{m}|{s}".encode()).hexdigest()[:12]
                                for p, m, s in zip(df.prompt_id, df.model_requested, df.setting)])
    bank = pd.read_csv(a.bank)
    df = df.merge(bank[["prompt_id", "domain", "task_type"]], on="prompt_id", how="left")
    df.to_csv(out / "passages.csv", index=False)
    with open(out / "raw_responses.jsonl", "w") as f:
        for r in raw:
            f.write(json.dumps(r, default=str) + "\n")
    print(len(df), "passages,", int(df.passage_ok.sum()), "pass the checks")
    print(df.groupby("label").agg(n=("text", "size"), ok=("passage_ok", "sum"), words=("n_words", "mean"),
                                  ratio=("n_words", lambda s: (s / df.loc[s.index, "target_words"]).mean()),
                                  cost=("cost_usd", "sum"), em=("has_em_dash", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
