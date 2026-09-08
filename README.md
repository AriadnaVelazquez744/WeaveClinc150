# WeaveClinc150 dataset

Pipeline **inspired by** [BlendX](https://arxiv.org/abs/2403.18277) (multi-intent mixing of single-intent corpora). This is **not** the official **BlendCLINC150** release; see [docs/DATASET_PROCEDURE.md](docs/DATASET_PROCEDURE.md) for the full methodology and required citations.

## Scripts

- `**generate_clinc150_multiintent.py**` — build multi-intent rows from CLINC150 `data_full.json`.
- `**rewrite_clinc150_multiintent.py**` — second pass: LLM rewrite (LM Studio).

## Documentation

- **[DATASET_PROCEDURE.md](docs/DATASET_PROCEDURE.md)** — complete construction procedure and relation to BlendX.

## Dependencies

```bash
python -m pip install -U numpy scikit-learn requests 
```

Optional: `sentence-transformers` for `--selection-method sentence_transformer`.

You may also install the virtual environment from [pyproject.toml](pyproject.toml)

## Generate (concat baseline, no HF quota)

```bash
python generate_clinc150_multiintent.py \
  --output-dir WeaveClinc150_dataset
```

Output includes `WeaveClinc150.json` (10k train / 2k val / 2k test, 150 in-domain intents, 2-3 blends per sample).

Alternative generated output with noise: `WeaveClinc150_dataset/WeaveClinc150_train_noisy.json`, `WeaveClinc150_dataset/WeaveClinc150_validation_noisy.json`, `WeaveClinc150_dataset/WeaveClinc150_test_noisy.json` (1-3 filler sentences per sample, using a noise pool of 19,698 programmatic statements).

## Rewrite pass (recommended: LM Studio)

```bash
python rewrite_clinc150_multiintent.py \
  --input-json WeaveClinc150_dataset/WeaveClinc150.json \
  --output-json WeaveClinc150_dataset/WeaveClinc150_rewritten.json
```

Configure LM Studio via `.env` (or set the same variables in your shell):

- `LMSTUDIO_MODEL`
- `LMSTUDIO_HOST`
- `LMSTUDIO_PORT`

## Other knobs

- Generator: `--train-size`, `--val-size`, `--test-size`, `--two-intent-ratio`, `--sim-min`, `--sim-max`, `--selection-method`, `--require-conjunction`, `--require-pronoun`
- Rewrite: `--strict`, `--max-rows`, resume is automatic (see DATASET_PROCEDURE.md)

## Dataset Files

| File | Description |
|------|-------------|
| `WeaveClinc150.json` | Phase A output: 10k train / 2k val / 2k test rows. Default `--two-intent-ratio: 0.5` yields ~50% k=2 and ~50% k=3 intent blends. Conjunction-required concatenations. |
| `WeaveClinc150_rewritten.json` | Phase B output: LLM‑rewritten text with `rewrite_model: "qwen2.5-7b-instruct"` (lowercase format). `was_rewritten: true` tracked, conjunction no longer required for success. |
| `WeaveClinc150_train_noisy.json` | Noisy training set: 10k rows with 1‑3 filler sentences per sample, drawn from a noise pool of 19,698 programmatic statements. |
| `WeaveClinc150_validation_noisy.json` | Noisy validation set: 2k rows with same noise injection. |
| `WeaveClinc150_test_noisy.json` | Noisy test set: 2k rows with same noise injection. |

**Dataset structure**

WeaveClinc150 provides 150 in-domain intents from CLINC150, with 2‑3 intents blended per sample across train/validation/test splits. The dataset comprises 14,000 synthetic multi‑intent samples generated through a concatenation‑first pipeline with similarity‑aware utterance selection (TF‑IDF, 1‑2 grams, L2‑normalized). Three fixed splits (10k/2k/2k) and 7,640 unique intent combinations provide a consistent testbed for evaluating models on intent detection tasks.

The noisy variants (`*_noisy.json`) were created by programmatically injecting 1‑3 filler sentences per sample from a pool of 19,698 programmatic statements, enabling robustness analysis for models exposed to conversational noise.

## Docker + Makefile

Use Docker as a wrapper for the full project lifecycle:

```bash
make docker-build
make start
```

Separated execution commands:

```bash
make generate
make rewrite-full
```

Quick smoke test:

```bash
make generate-smoke