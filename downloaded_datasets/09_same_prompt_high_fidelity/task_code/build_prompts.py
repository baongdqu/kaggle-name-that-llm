"""Build my prompt bank from topics.py: one prompt per topic, in the host's register.

Prompts that come too close to a train topic or to another public bank are dropped (see overlap.py).
Usage: build_prompts.py <research dir> <out csv>
"""
import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from topics import TOPICS

STYLES = [
    "vividly, for a curious general reader",
    "using one extended metaphor",
    "as if to a bright twelve-year-old",
    "in a lyrical, almost poetic register",
    "with a light touch of humour",
    "in the voice of a popular science writer",
    "in a quiet, reflective tone",
    "concretely, anchored in one everyday example",
    "with a sense of wonder",
    "plainly and precisely, without jargon",
]
OPENINGS = ["Imagine", "Picture", "At first glance,", "Every", "Long before", "There is a moment when", "Most people never notice"]
TASKS = {
    "explain": "Explain {topic}, {style}.",
    "describe": "Describe {topic}, {style}.",
    "argue": "Make the case for one side of {topic}, {style}.",
    "story": "Tell a very short story about {topic}, {style}.",
    "evoke": "Write an evocative passage about {topic}, {style}.",
}


def task_for(domain, rng):
    if domain == "creative prose":
        return rng.choices(["evoke", "story", "describe"], [5, 3, 2])[0]
    if domain == "philosophy":
        return rng.choices(["explain", "argue", "describe"], [5, 3, 1])[0]
    if domain == "history":
        return rng.choices(["explain", "describe"], [5, 4])[0]
    return rng.choices(["explain", "describe"], [8, 2])[0]


def main(research, out):
    rng = random.Random(20261018)
    rows = []
    for domain, topics in TOPICS.items():
        for topic in topics:
            task = task_for(domain, rng)
            style = rng.choice(STYLES)
            body = TASKS[task].format(topic=topic, style=style)
            opening = rng.choice(OPENINGS) if rng.random() < 0.25 else ""
            if opening:
                body += f" Begin with the words '{opening}'."
            rows.append(dict(domain=domain, task_type=task, topic=topic, style=style, opening=opening, prompt_body=body))
    df = pd.DataFrame(rows)

    r = Path(research)
    others = list(pd.read_csv(r / "dl/pjm/train_prompts_reconstructed.csv").inferred_prompt)
    n_train = len(others)
    others += list(pd.read_csv(r / "dl/seyit/prompts.csv").topic.astype(str))
    others += list(pd.read_csv(r / "dl/prv/prompts.csv").prompt.astype(str))
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).fit(list(df.topic) + others)
    S = cosine_similarity(vec.transform(df.topic), vec.transform(others))
    df["max_sim_train"] = S[:, :n_train].max(1).round(3)
    df["max_sim_public"] = S[:, n_train:].max(1).round(3)
    keep = (df.max_sim_train < 0.3) & (df.max_sim_public < 0.45)
    print(f"{len(df)} prompts, dropped {(~keep).sum()} for overlap")
    df = df[keep].reset_index(drop=True)
    df.insert(0, "prompt_id", [f"h{i:04d}" for i in range(len(df))])
    df.to_csv(out, index=False)
    print(df.domain.value_counts().to_dict())
    print(df.task_type.value_counts().to_dict())
    print(df.sample(5, random_state=0).prompt_body.tolist())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
