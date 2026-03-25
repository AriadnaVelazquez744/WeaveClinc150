#!/usr/bin/env python3
"""
Rewrite an existing WeaveClinc150 dataset (see docs/DATASET_PROCEDURE.md) with an LLM.

Backend:
- lmstudio: local LM Studio OpenAI-compatible server (OpenAI-compatible `/v1/chat/completions`).

This script keeps the original dataset structure and fields, and only updates:
- text
- metadata.was_rewritten
- metadata.rewrite_model

Everything else (labels, source_texts, source_intents, split membership, etc.)
is preserved to maintain the same base dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import requests

# Official Hub id for the Qwen2.5-7B instruct checkpoint (matches src/base_models/cache/models--Qwen--Qwen2.5-7B-Instruct/).
DEFAULT_QWEN25_7B_INSTRUCT_ID = "Qwen/Qwen2.5-7B-Instruct"

def _load_dotenv_file(dotenv_path: Path) -> None:
    """
    Minimal `.env` loader (no external dependency).
    - Supports KEY=VALUE lines
    - Ignores blank lines and comments starting with '#'
    - Does not override existing environment variables
    """
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if key in os.environ:
            continue
        # Strip simple surrounding quotes.
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply LLM rewrite over WeaveClinc150 dataset JSON")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("WeaveClinc150_dataset/WeaveClinc150.json"),
        help="Input WeaveClinc150 dataset JSON path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("WeaveClinc150_dataset/WeaveClinc150_rewritten.json"),
        help="Output rewritten dataset JSON path",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--require-conjunction",
        action="store_true",
        default=True,
        help="Keep only rewrites containing conjunction markers; fallback if missing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any row cannot be rewritten",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap for quick runs/testing (0 means all rows)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=True,
        help="Validate LM Studio connectivity before processing all rows",
    )
    parser.add_argument(
        "--lmstudio-base-url",
        type=str,
        default=os.getenv("LMSTUDIO_BASE_URL", ""),
        help="Optional full LM Studio base URL (must include /v1). If empty, computed from host+port.",
    )
    parser.add_argument(
        "--lmstudio-host",
        type=str,
        default=os.getenv("LMSTUDIO_HOST", "127.0.0.1"),
        help="LM Studio host/address (env: LMSTUDIO_HOST)",
    )
    parser.add_argument(
        "--lmstudio-port",
        type=int,
        default=int(os.getenv("LMSTUDIO_PORT", "1234")),
        help="LM Studio port (env: LMSTUDIO_PORT)",
    )
    parser.add_argument(
        "--lmstudio-api-key",
        type=str,
        default=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        help="Dummy API key (LM Studio accepts any non-empty string)",
    )
    parser.add_argument(
        "--lmstudio-model",
        type=str,
        default=os.getenv("LMSTUDIO_MODEL", DEFAULT_QWEN25_7B_INSTRUCT_ID),
        help=(
            "Model identifier exactly as shown in LM Studio when the model is loaded "
            f"(often '{DEFAULT_QWEN25_7B_INSTRUCT_ID}' for Qwen2.5-7B-Instruct)"
        ),
    )
    parser.add_argument(
        "--lmstudio-timeout",
        type=int,
        default=300,
        help="HTTP timeout seconds per LM Studio request",
    )
    return parser.parse_args()


CONJUNCTIONS = (" and ", " also ", " plus ", " as well as ", " then ")
SPLITS = ("train", "validation", "test")


def normalize_space(text: str) -> str:
    return " ".join(text.strip().split())


def contains_conjunction(text: str) -> bool:
    lower = f" {text.lower()} "
    return any(c in lower for c in CONJUNCTIONS)


def row_signature(row: dict[str, Any]) -> str:
    """
    Stable identity for matching processed rows against input rows when resuming.
    Prefer source utterances (immutable input provenance), then concatenated text.
    """
    source_texts = row.get("source_texts")
    if isinstance(source_texts, list) and source_texts:
        key_text = " || ".join(normalize_space(str(x)) for x in source_texts)
    else:
        meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        key_text = normalize_space(str(meta.get("concatenated_text", row.get("text", ""))))
    labels = row.get("labels", [])
    key_labels = "|".join(sorted(str(x) for x in labels)) if isinstance(labels, list) else str(labels)
    return f"{key_labels}##{key_text}"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def init_or_load_output(input_data: dict[str, list[dict[str, Any]]], output_path: Path) -> dict[str, list[dict[str, Any]]]:
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        for split in SPLITS:
            val = loaded.get(split, [])
            if not isinstance(val, list):
                raise RuntimeError(f"Invalid output JSON format: '{split}' must be a list.")
        return loaded

    # Create output scaffold once so interrupted runs always leave a resumable file.
    created = {split: [] for split in SPLITS}
    atomic_write_json(output_path, created)
    return created


def row_rewrite_succeeded(row: dict[str, Any]) -> bool:
    meta = row.get("metadata", {})
    if not isinstance(meta, dict):
        return False
    return meta.get("was_rewritten") is True


def count_rows_ready(input_data: dict[str, Any], output_data: dict[str, Any]) -> dict[str, int]:
    """How many rows per split already have a successful rewrite that aligns with input (any index)."""
    counts: dict[str, int] = {}
    for split in SPLITS:
        in_rows = input_data.get(split, [])
        out_rows = output_data.get(split, [])
        total_ok = 0
        for i, in_row in enumerate(in_rows):
            if i >= len(out_rows):
                break
            o = out_rows[i]
            if row_signature(o) == row_signature(in_row) and row_rewrite_succeeded(o):
                total_ok += 1
        counts[split] = total_ok
    return counts


def first_pending_index(in_rows: list[dict[str, Any]], out_rows: list[dict[str, Any]]) -> int:
    """
    Return the first index that is not yet successfully rewritten and aligned with input.
    This lets resume continue from the last processed point.
    """
    limit = min(len(in_rows), len(out_rows))
    i = 0
    while i < limit:
        o = out_rows[i]
        if row_signature(o) == row_signature(in_rows[i]) and row_rewrite_succeeded(o):
            i += 1
            continue
        break
    return i


def build_prompt(source_texts: list[str], source_intents: list[str]) -> str:
    utterances = "\n".join(f"{i + 1}. {u}" for i, u in enumerate(source_texts))
    intent_str = ", ".join(source_intents)
    return (
        "You are rewriting a combined multi-intent user query.\n"
        "Rules:\n"
        "1) Preserve ALL intents and meaning.\n"
        "2) Make it sound natural and conversational.\n"
        "3) Keep it as ONE user utterance.\n"
        "4) Include a conjunction (and/also/plus/then/as well as).\n"
        "5) Output only the rewritten utterance.\n"
        f"Intents: {intent_str}\n"
        f"Source utterances:\n{utterances}\n"
    )


def lmstudio_chat_completions_url(base_url: str) -> str:
    b = base_url.rstrip("/")
    if not b.endswith("/v1"):
        b = f"{b}/v1"
    return f"{b}/chat/completions"


def build_lmstudio_base_url(host: str, port: int, base_url_override: str) -> str:
    """
    Base URL used by lmstudio_chat_completions_url.

    Precedence:
    1) --lmstudio-base-url / env `LMSTUDIO_BASE_URL` if provided
    2) host+port (env/CLI) otherwise
    """
    if base_url_override and base_url_override.strip():
        return base_url_override.strip()
    return f"http://{host}:{port}/v1"


def rewrite_text_lmstudio(
    base_url: str,
    api_key: str,
    model: str,
    source_texts: list[str],
    source_intents: list[str],
    max_new_tokens: int,
    temperature: float,
    timeout: int,
) -> tuple[str | None, str | None]:
    prompt = build_prompt(source_texts, source_intents)
    url = lmstudio_chat_completions_url(base_url)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You rewrite task-oriented user utterances."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return None, "LM Studio: empty choices in response"
        msg = (choices[0].get("message") or {}).get("content") or ""
        text = normalize_space(str(msg).strip())
        return (text if text else None), None
    except requests.exceptions.RequestException as exc:
        return None, f"LM Studio: {type(exc).__name__}: {exc}"
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return None, f"LM Studio: bad response JSON: {type(exc).__name__}: {exc}"


def rewrite_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    primary_model_label: str,
    max_new_tokens: int,
    temperature: float,
    require_conjunction: bool,
) -> tuple[dict[str, Any], bool, str | None]:
    new_row = deepcopy(row)
    metadata = dict(new_row.get("metadata", {}))

    source_texts = new_row.get("source_texts")
    if not isinstance(source_texts, list) or not source_texts:
        concat_text = metadata.get("concatenated_text", new_row.get("text", ""))
        source_texts = [concat_text] if concat_text else []
    source_intents = new_row.get("source_intents", new_row.get("labels", []))

    rewritten = rewrite_text_lmstudio(
        base_url=args.lmstudio_base_url,
        api_key=args.lmstudio_api_key,
        model=args.lmstudio_model,
        source_texts=source_texts,
        source_intents=source_intents,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        timeout=args.lmstudio_timeout,
    )
    rewritten_text, err = rewritten

    if rewritten_text is None:
        metadata["was_rewritten"] = False
        metadata["rewrite_model"] = primary_model_label
        new_row["metadata"] = metadata
        return new_row, False, err

    if require_conjunction and not contains_conjunction(rewritten_text):
        metadata["was_rewritten"] = False
        metadata["rewrite_model"] = primary_model_label
        new_row["metadata"] = metadata
        return new_row, False, "Rewrite produced text without required conjunction."

    if rewritten_text == normalize_space(new_row.get("text", "")):
        metadata["was_rewritten"] = False
        metadata["rewrite_model"] = primary_model_label
        new_row["metadata"] = metadata
        return new_row, False, "Rewrite matched original text."

    new_row["text"] = rewritten_text
    metadata["was_rewritten"] = True
    metadata["rewrite_model"] = primary_model_label
    new_row["metadata"] = metadata
    return new_row, True, None


def probe_lmstudio(args: argparse.Namespace) -> None:
    prompt = "Rewrite into one natural sentence with conjunction: check my balance and transfer money."
    out, err = rewrite_text_lmstudio(
        base_url=args.lmstudio_base_url,
        api_key=args.lmstudio_api_key,
        model=args.lmstudio_model,
        source_texts=[prompt],
        source_intents=["check_balance", "transfer_money"],
        max_new_tokens=24,
        temperature=0.2,
        timeout=min(120, args.lmstudio_timeout),
    )
    if not out:
        raise RuntimeError(err or "LM Studio probe failed with no output.")


def main() -> int:
    # Load repository `.env` so argparse defaults can come from it.
    _load_dotenv_file(Path(".env"))
    args = parse_args()
    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    args.lmstudio_base_url = build_lmstudio_base_url(args.lmstudio_host, args.lmstudio_port, args.lmstudio_base_url)
    # Keep same metadata style as current rewritten dataset (model id only).
    primary_model_label = args.lmstudio_model
    if args.fail_fast:
        try:
            probe_lmstudio(args)
        except Exception as exc:
            raise RuntimeError(
                f"Inference probe failed before processing dataset (lmstudio): {type(exc).__name__}: {exc}"
            ) from exc
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    had_output_file = args.output_json.exists()
    out_data = init_or_load_output(data, args.output_json)
    already_ok = count_rows_ready(data, out_data)
    stats: dict[str, Any] = {
        "api_calls_this_run": 0,
        "rows_reused_already_ok": 0,
        "rewritten_rows_this_run": 0,
        "failed_rows_this_run": 0,
        "sample_failures": [],
        "resumed": had_output_file,
        "rows_already_ok_by_split_start": already_ok,
    }

    remaining = args.max_rows if args.max_rows > 0 else None
    for split in SPLITS:
        in_rows = data.get(split, [])
        prev_out = out_data.get(split, [])
        if not isinstance(prev_out, list):
            prev_out = []

        start_idx = first_pending_index(in_rows, prev_out)
        merged: list[dict[str, Any]] = [deepcopy(prev_out[i]) for i in range(start_idx)]

        for i in range(start_idx, len(in_rows)):
            in_row = in_rows[i]
            # Compare to existing output row from last saved file / earlier splits.
            existing = prev_out[i] if i < len(prev_out) else None

            if (
                existing is not None
                and row_signature(existing) == row_signature(in_row)
                and row_rewrite_succeeded(existing)
            ):
                merged.append(deepcopy(existing))
                stats["rows_reused_already_ok"] += 1
                continue

            if remaining is not None and remaining <= 0:
                merged.append(deepcopy(in_row))
                continue

            new_row, ok, err = rewrite_row(
                row=in_row,
                args=args,
                primary_model_label=primary_model_label,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                require_conjunction=args.require_conjunction,
            )
            stats["api_calls_this_run"] += 1
            merged.append(new_row)
            out_data[split] = merged + [deepcopy(in_rows[j]) for j in range(i + 1, len(in_rows))]
            # Temporarily pad tail with input copies so file length matches input during checkpoint
            atomic_write_json(args.output_json, out_data)
            out_data[split] = merged

            if ok:
                stats["rewritten_rows_this_run"] += 1
            else:
                stats["failed_rows_this_run"] += 1
                if len(stats["sample_failures"]) < 5:
                    stats["sample_failures"].append(err or "Unknown rewrite failure")

            if remaining is not None:
                remaining -= 1

        # Final split: full length, no padding duplicates
        while len(merged) < len(in_rows):
            merged.append(deepcopy(in_rows[len(merged)]))
        out_data[split] = merged[: len(in_rows)]
        atomic_write_json(args.output_json, out_data)

        if remaining is not None and remaining <= 0:
            break

    atomic_write_json(args.output_json, out_data)

    total_in = sum(len(data.get(s, [])) for s in SPLITS)
    total_ok_now = sum(count_rows_ready(data, out_data).values())
    stats["total_input_rows"] = total_in
    stats["total_successfully_rewritten_in_output"] = total_ok_now
    stats["rows_still_pending_rewrite"] = total_in - total_ok_now

    if args.strict and stats["rows_still_pending_rewrite"] > 0:
        raise RuntimeError(
            f"Strict mode failed: {stats['rows_still_pending_rewrite']} rows still lack a successful rewrite."
        )

    stats["rewrite_rate_pct"] = round((total_ok_now * 100.0 / total_in) if total_in else 0.0, 4)
    stats["input_json"] = str(args.input_json)
    stats["output_json"] = str(args.output_json)
    stats["backend"] = "lmstudio"
    stats["lmstudio_base_url"] = args.lmstudio_base_url
    stats["lmstudio_model"] = args.lmstudio_model
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
