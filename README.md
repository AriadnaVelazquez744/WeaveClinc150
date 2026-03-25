# WeaveClinc150 dataset

Pipeline **inspired by** [BlendX](https://arxiv.org/abs/2403.18277) (multi-intent mixing of single-intent corpora). This is **not** the official **BlendCLINC150** release; see [docs/DATASET_PROCEDURE.md](docs/DATASET_PROCEDURE.md) for the full methodology and required citations.

## Scripts

- `**generate_clinc150_multiintent.py`** — build multi-intent rows from CLINC150 `data_full.json`.
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

Output includes `WeaveClinc150.json`.

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

