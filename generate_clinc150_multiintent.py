#!/usr/bin/env python3
"""
Generate the WeaveClinc150 dataset (BlendX-inspired; not the official BlendCLINC150 release).

See docs/DATASET_PROCEDURE.md for methodology and citation of the original BlendX / BlendCLINC150 work.

Features:
- Drops OOD intents and keeps split integrity (train/val/test).
- Builds 2-intent and 3-intent combinations with configurable ratios.
- Similarity-aware utterance selection (TF-IDF or sentence embeddings).
- Rule-based concatenation templates.
- Conjunction-based concatenation only (no LLM rewrite).
- Quality filtering (length, conjunction, pronoun).
- Exports:
  - WeaveClinc150.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


SPLIT_NAME_MAP = {"train": "train", "val": "validation", "test": "test"}
CONJUNCTIONS = (" and ", " also ", " plus ", " as well as ", " then ")
PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
    "we",
    "our",
    "ours",
}


@dataclass
class Example:
    text: str
    intent: str


@dataclass
class SplitStore:
    examples: list[Example]
    by_intent: dict[str, list[int]]
    embeddings: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WeaveClinc150 dataset (BlendX-inspired)")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("clinc150/data_full.json"),
        help="Path to CLINC150 data_full.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("WeaveClinc150_dataset"),
        help="Output directory",
    )
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument(
        "--two-intent-ratio",
        type=float,
        default=0.5,
        help="Probability of sampling 2 intents, else 3 intents",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-method",
        type=str,
        choices=["tfidf", "sentence_transformer"],
        default="tfidf",
        help="Similarity backend used for source utterance selection",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence-transformers model (used only with --selection-method sentence_transformer)",
    )
    parser.add_argument(
        "--sim-min",
        type=float,
        default=0.0,
        help="Minimum average cosine similarity for selected source utterances",
    )
    parser.add_argument(
        "--sim-max",
        type=float,
        default=0.85,
        help="Maximum average cosine similarity for selected source utterances",
    )
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=45)
    parser.add_argument(
        "--require-conjunction",
        action="store_true",
        default=True,
        help="Require conjunction marker in final utterance",
    )
    parser.add_argument(
        "--require-pronoun",
        action="store_true",
        default=False,
        help="Require conversational pronoun in final utterance",
    )
    parser.add_argument(
        "--max-attempts-per-item",
        type=int,
        default=40,
        help="Sampling retries before forcing fallback acceptance",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join(text.strip().split())


def load_clinc_in_domain(path: Path) -> dict[str, list[Example]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    out: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        for item in raw[split]:
            text, intent = item
            if intent.lower().startswith("oos"):
                continue
            out[split].append(Example(text=normalize_space(text), intent=intent))
    return out


def build_split_store(split_examples: list[Example], embeddings: np.ndarray) -> SplitStore:
    texts = [x.text for x in split_examples]
    by_intent: dict[str, list[int]] = {}
    for i, ex in enumerate(split_examples):
        by_intent.setdefault(ex.intent, []).append(i)
    return SplitStore(
        examples=split_examples,
        by_intent=by_intent,
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def build_embeddings(split_examples: list[Example], selection_method: str, embedding_model: str) -> np.ndarray:
    texts = [x.text for x in split_examples]
    if selection_method == "tfidf":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        mat = vec.fit_transform(texts)
        dense = mat.astype(np.float32).toarray()
        return normalize(dense, norm="l2", axis=1)
    if selection_method == "sentence_transformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for --selection-method sentence_transformer. "
                "Install with: pip install sentence-transformers"
            ) from exc
        embedder = SentenceTransformer(embedding_model)
        return np.asarray(
            embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32
        )
    raise ValueError(f"Unsupported selection method: {selection_method}")


def avg_pairwise_cosine(embs: np.ndarray) -> float:
    if len(embs) < 2:
        return 1.0
    sim_sum = 0.0
    pairs = 0
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sim_sum += float(np.dot(embs[i], embs[j]))
            pairs += 1
    return sim_sum / max(1, pairs)


def choose_templates(k: int, rng: random.Random) -> str:
    if k == 2:
        return rng.choice(
            [
                "{u1} and {u2}",
                "{u1}, and also {u2}",
                "{u1}. Also, {u2}.",
                "Can you {u1} and then {u2}?",
            ]
        )
    return rng.choice(
        [
            "{u1}, {u2}, and {u3}",
            "{u1}; also {u2}; and {u3}",
            "Please {u1}, then {u2}, and also {u3}",
            "{u1}. Also {u2}, and {u3}.",
        ]
    )


def build_concat(source_texts: list[str], rng: random.Random) -> str:
    t = choose_templates(len(source_texts), rng)
    mapping = {f"u{i+1}": source_texts[i] for i in range(len(source_texts))}
    return normalize_space(t.format(**mapping))


def contains_conjunction(text: str) -> bool:
    lower = f" {text.lower()} "
    return any(c in lower for c in CONJUNCTIONS)


def contains_pronoun(text: str) -> bool:
    tokens = [t.strip(".,!?;:'\"()[]{}").lower() for t in text.split()]
    return any(t in PRONOUNS for t in tokens)


def passes_quality(
    text: str,
    min_words: int,
    max_words: int,
    require_conjunction: bool,
    require_pronoun: bool,
) -> bool:
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if require_conjunction and not contains_conjunction(text):
        return False
    if require_pronoun and not contains_pronoun(text):
        return False
    return True


def sample_candidate_indices(
    store: SplitStore,
    intents: list[str],
    rng: random.Random,
) -> list[int]:
    idxs = []
    for intent in intents:
        idxs.append(rng.choice(store.by_intent[intent]))
    return idxs


def generate_split(
    split_name: str,
    target_size: int,
    store: SplitStore,
    rng: random.Random,
    two_intent_ratio: float,
    sim_min: float,
    sim_max: float,
    min_words: int,
    max_words: int,
    require_conjunction: bool,
    require_pronoun: bool,
    max_attempts_per_item: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    intents_all = sorted(store.by_intent.keys())
    rows: list[dict[str, Any]] = []
    for _ in range(target_size):
        accepted = None
        for attempt in range(max_attempts_per_item):
            blend_size = 2 if rng.random() < two_intent_ratio else 3
            sampled_intents = rng.sample(intents_all, k=blend_size)
            idxs = sample_candidate_indices(store, sampled_intents, rng)
            embs = store.embeddings[idxs]
            avg_sim = avg_pairwise_cosine(embs)
            if not (sim_min <= avg_sim <= sim_max):
                continue
            source_texts = [store.examples[i].text for i in idxs]
            concat_text = build_concat(source_texts, rng)
            final_text = normalize_space(concat_text)
            if not passes_quality(
                final_text,
                min_words=min_words,
                max_words=max_words,
                require_conjunction=require_conjunction,
                require_pronoun=require_pronoun,
            ):
                if attempt < max_attempts_per_item - 1:
                    continue
            accepted = {
                "text": final_text,
                "labels": sampled_intents,
                "source_intents": sampled_intents,
                "source_texts": source_texts,
                "metadata": {
                    "split": split_name,
                    "blend_size": blend_size,
                    "avg_source_cosine_similarity": avg_sim,
                    "was_rewritten": False,
                    "concatenated_text": concat_text,
                },
            }
            break
        if accepted is None:
            # Defensive fallback if constraints are too strict.
            sampled_intents = rng.sample(intents_all, k=2)
            idxs = sample_candidate_indices(store, sampled_intents, rng)
            source_texts = [store.examples[i].text for i in idxs]
            concat_text = build_concat(source_texts, rng)
            accepted = {
                "text": concat_text,
                "labels": sampled_intents,
                "source_intents": sampled_intents,
                "source_texts": source_texts,
                "metadata": {
                    "split": split_name,
                    "blend_size": 2,
                    "avg_source_cosine_similarity": avg_pairwise_cosine(store.embeddings[idxs]),
                    "was_rewritten": False,
                    "concatenated_text": concat_text,
                },
            }
        rows.append(accepted)
    return rows, {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    split_examples = load_clinc_in_domain(args.input_json)
    stores = {}
    for split, exs in split_examples.items():
        embs = build_embeddings(exs, args.selection_method, args.embedding_model)
        stores[split] = build_split_store(exs, embs)

    sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    output_json: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        output_rows, split_stats = generate_split(
            split_name=split,
            target_size=sizes[split],
            store=stores[split],
            rng=rng,
            two_intent_ratio=args.two_intent_ratio,
            sim_min=args.sim_min,
            sim_max=args.sim_max,
            min_words=args.min_words,
            max_words=args.max_words,
            require_conjunction=args.require_conjunction,
            require_pronoun=args.require_pronoun,
            max_attempts_per_item=args.max_attempts_per_item,
        )
        output_json[SPLIT_NAME_MAP[split]] = output_rows

    json_path = args.output_dir / "WeaveClinc150.json"
    write_json(json_path, output_json)

    summary = {
        "output_json": str(json_path),
        "sizes": {k: len(v) for k, v in output_json.items()},
        "rewrite_enabled": False,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
