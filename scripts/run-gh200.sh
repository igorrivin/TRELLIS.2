#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run-gh200.sh PORT [--share]

Runs TRELLIS.2 in the GH200 Docker image and forwards PORT on the host to the
same port in the container.

Options:
  --share       Ask Gradio to create a public share URL.
  --image NAME  Docker image to run. Defaults to trellis2-gh200:latest.
  -h, --help    Show this help.

Environment:
  HF_CACHE      Host Hugging Face cache directory. Defaults to ~/.cache/huggingface.
USAGE
}

port=""
share_args=()
image="${TRELLIS2_IMAGE:-trellis2-gh200:latest}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --share)
      share_args+=(--share)
      shift
      ;;
    --image)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --image" >&2
        exit 2
      fi
      image="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$port" ]]; then
        echo "Unexpected extra argument: $1" >&2
        usage >&2
        exit 2
      fi
      port="$1"
      shift
      ;;
  esac
done

if [[ -z "$port" ]]; then
  usage >&2
  exit 2
fi

if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
  echo "PORT must be an integer between 1 and 65535." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hf_cache="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$hf_cache"

exec docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "${port}:${port}" \
  -v "${repo_root}:/opt/TRELLIS.2" \
  -v "${hf_cache}:/root/.cache/huggingface" \
  -w /opt/TRELLIS.2 \
  "${image}" \
  python app.py --server-name 0.0.0.0 --server-port "${port}" "${share_args[@]}"
