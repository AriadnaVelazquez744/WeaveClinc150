#!/usr/bin/env bash
set -euo pipefail

mode="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${mode}" in
  generate)
    exec python /app/generate_clinc150_multiintent.py "$@"
    ;;
  rewrite)
    exec python /app/rewrite_clinc150_multiintent.py "$@"
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage:
  docker run ... <image> generate [args...]
  docker run ... <image> rewrite [args...]

Examples:
  docker run --rm -v "$PWD":/app <image> generate --output-dir WeaveClinc150_dataset
  docker run --rm --env-file .env -v "$PWD":/app <image> rewrite --input-json WeaveClinc150_dataset/WeaveClinc150.json
EOF
    ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    exit 2
    ;;
esac

