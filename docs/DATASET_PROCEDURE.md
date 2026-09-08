# WeaveClinc150: full construction procedure

This document describes how the **WeaveClinc150** corpus in this repository is built. The pipeline is **inspired by** the BlendX family (combining single-intent CLINC150 utterances into multi-label examples) but it is **not** the official **BlendCLINC150** release from the BlendX authors. Cite both this procedure and the original work when you use the data (see section 6).

---

## 1. Goals

- **Multi-label** examples: each utterance has a **set** of CLINC150 in-domain intents (2 or 3 distinct intents per row).
- **Split discipline**: blends for `train` use only CLINC150 **train** sentences (same for `val` and `test`), no cross-split leakage of source utterances.
- **Source**: UCI `data_full.json` (same content as `clinc/oos-eval`).
- **Concatenation-first** utterance, then optional **LLM paraphrase**, with **checkpointed** rewriting.

---

## 2. Source data

- **File**: `data_full.json` (CLINC150 UCI or `oos-eval`).
- **Splits**: `train`, `val`, `test`.
- **Filter**: remove OOD rows (`intent.lower().startswith("oos")`).
- **Result**: 150 in-domain intents only.

---

## 3. Phase A: synthetic rows (`generate_clinc150_multiintent.py`)

### 3.1 Intent sampling

Per target row (defaults: 10000 train, 2000 val, 2000 test):

1. With probability `two_intent_ratio` (default 0.5): **2** intents, else **3**.
2. Sample distinct intent names from intents present in that split.

### 3.2 Utterance selection (similarity-aware)

For each intent, sample one utterance from that intent **in the same split**.

Per split, embed all in-domain texts:

- **Default**: TF-IDF (1-2 grams), L2-normalized; **average pairwise cosine** across selected utterances.
- **Optional**: `sentence_transformers` (`--selection-method sentence_transformer`).

Accept candidate if average pairwise cosine is in `[sim_min, sim_max]` (defaults 0.0 to 0.85). After `max_attempts_per_item` failures, a **fallback** 2-intent concatenation row is emitted (similarity window not re-checked).

### 3.3 Concatenation templates

Random templates, e.g. `{u1} and {u2}`, `{u1}. Also, {u2}.`, three-intent variants with `and` / `also` / `then`.

Stored as `text`, `metadata.concatenated_text`, `labels`, `source_intents`, `source_texts`.

**Note**: `metadata.blend_size` is the number of combined intents (2 or 3); the key name is historical.

### 3.4 Quality filters

- Word count in `[min_words, max_words]` (defaults 6 to 45).
- If `require_conjunction` (default on): substring check for connectives (`and`, `also`, `plus`, `as well as`, `then`).
- Optional `require_pronoun`.

### 3.5 Exports

- `WeaveClinc150.json`: keys `train`, `validation`, `test`.

---

## 4. Phase B: LLM rewrite (`rewrite_clinc150_multiintent.py`)

- Preserves `labels`, `source_texts`, `source_intents`.
- Sets `metadata.was_rewritten` when rewrite succeeds (text change).
- **LM Studio** (default): OpenAI-compatible chat at `http://127.0.0.1:1234/v1` (configurable via `.env` host/port/model).
- **Resume**: skips rows with matching signature and `was_rewritten: true`; checkpoints after each call.

---

## 5. Output schema (per row)

| Field                                   | Meaning             |
| --------------------------------------- | ------------------- |
| `text`                                  | Final utterance.    |
| `labels`                                | Intent set.         |
| `source_intents` / `source_texts`       | Provenance.         |
| `metadata.split`                        | train / val / test. |
| `metadata.blend_size`                   | 2 or 3 intents.     |
| `metadata.avg_source_cosine_similarity` | Phase A similarity. |
| `metadata.concatenated_text`            | Pre-rewrite concat. |
| `metadata.was_rewritten`                | Phase B success.    |
| `metadata.rewrite_model`                | Model id used.      |

---

## 5.1 Dataset files

This repository provides the following dataset files in `WeaveClinc150_dataset/`:

| File | Description |
|------|-------------|
| `WeaveClinc150.json` | Phase A output: 10k train / 2k val / 2k test rows. Default `--two-intent-ratio: 0.5` yields ~50% k=2 and ~50% k=3 intent blends. All concatenations include a required conjunction marker (Phase A default). |
| `WeaveClinc150_rewritten.json` | Phase B output: LLM‑rewritten text with `rewrite_model: "qwen2.5-7b-instruct"` (lowercase). `was_rewritten: true` tracked, but conjunction no longer required for success. |
| `WeaveClinc150_train_noisy.json` | Noisy training set: 10k rows with 1‑3 filler sentences per sample, drawn from a noise pool of 19,698 programmatic statements. |
| `WeaveClinc150_validation_noisy.json` | Noisy validation set: 2k rows with same noise injection. |
| `WeaveClinc150_test_noisy.json` | Noisy test set: 2k rows with same noise injection. |

**Dataset structure**

WeaveClinc150 provides 150 in-domain intents from CLINC150, with 2‑3 intents blended per sample across train/validation/test splits. The dataset comprises 14,000 synthetic multi‑intent samples generated through a concatenation‑first pipeline with similarity‑aware utterance selection (TF‑IDF, 1‑2 grams, L2‑normalized). Three fixed splits (10k/2k/2k) and 7,640 unique intent combinations provide a consistent testbed for evaluating models on intent detection tasks.

The noisy variants (`*_noisy.json`) were created by programmatically injecting 1‑3 filler sentences per sample from a pool of 19,698 programmatic statements, enabling robustness analysis for models exposed to conversational noise.

---

## 6. Relation to BlendX / BlendCLINC150

**BlendX** (Li et al.) combines single-intent corpora (including CLINC150), uses concatenation-style mixing and LLM rewriting with similarity-driven selection, and releases **BlendCLINC150** as part of BlendX.

This implementation follows the **same high-level recipe** (in-domain CLINC150, 2 to 3 intents, similarity gate, concat plus LLM) but differs in templates, default similarity (TF-IDF), prompts, sizes, and filters from the paper and from the official **HYU-NLP/BlendX** dataset files.

**Name the corpus** as WeaveClinc150 (BlendX-**inspired**), not as official BlendCLINC150 unless you use their release.

Key differences from BlendX/BlendCLINC150 in this repository:

- Phase A is implemented as conjunction‑based synthetic concatenations with the same split discipline, but without the generator-internal LLM rewrite (rewrite is done only in Phase B).
- Similarity-aware utterance selection uses a TF‑IDF (1‑2 grams) + L2‑normalized default and a configurable cosine gate; templates and filter heuristics may differ from the original releases.
- Phase B rewrite uses a local LM Studio OpenAI‑compatible endpoint and a checkpointed "rewrite-or-skip" resume strategy, rather than relying on the specific public BlendX rewriting setup.

---