#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run-gh200.sh PORT [--share] [--rodin-key KEY] [--3daistudio-key KEY]

Runs TRELLIS.2 in the GH200 Docker image and forwards PORT on the host to the
same port in the container.

Options:
  --share          Ask Gradio to create a public share URL.
  --rodin-key KEY  Pass a Hyper3D/Rodin API key into the container.
  --3daistudio-key KEY
                   Pass a 3D AI Studio API key into the container.
  --image NAME     Docker image to run. Defaults to trellis2-gh200:latest.
  -h, --help       Show this help.

Environment:
  HF_CACHE           Host Hugging Face cache directory. Defaults to ~/.cache/huggingface.
  .env               Optional repo-root environment file loaded before launch.
  HYPER3D_API_KEY    Hyper3D/Rodin API key. Used if --rodin-key is omitted.
  RODIN_API_KEY      Alternative Rodin API key env var.
  HITEM3D_CLIENT_ID  Hitem3D public client id.
  HITEM3D_CLIENT_SECRET
                     Hitem3D client secret.
  THREEDAI_API_KEY   3D AI Studio API key for Tripo P1.
  THREEDAISTUDIO_API_KEY, AI3DSTUDIO_API_KEY, THREEDAI_STUDIO_API_KEY, 3DAISTUDIO_API_KEY
                     Alternative 3D AI Studio API key env vars.
USAGE
}

strip_dotenv_quotes() {
  local value="$1"
  value="${value%$'\r'}"
  if [[ "$value" == \"*\" && "$value" == *\" ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi
  printf '%s' "$value"
}

dotenv_value() {
  local key="$1"
  local line name value
  [[ -f "${repo_root}/.env" ]] || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"
    name="${line%%=*}"
    value="${line#*=}"
    name="${name%"${name##*[![:space:]]}"}"
    if [[ "$name" == "$key" && "$line" == *"="* ]]; then
      strip_dotenv_quotes "$value"
      return 0
    fi
  done < "${repo_root}/.env"
  return 1
}

load_dotenv() {
  local line name value
  [[ -f "${repo_root}/.env" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"
    name="${line%%=*}"
    value="${line#*=}"
    name="${name%"${name##*[![:space:]]}"}"
    if [[ "$line" == *"="* && "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      export "${name}=$(strip_dotenv_quotes "$value")"
    fi
  done < "${repo_root}/.env"
}

port=""
share_args=()
image="${TRELLIS2_IMAGE:-trellis2-gh200:latest}"
rodin_key=""
threedai_key=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --share)
      share_args+=(--share)
      shift
      ;;
    --rodin-key)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --rodin-key" >&2
        exit 2
      fi
      rodin_key="$2"
      shift 2
      ;;
    --3daistudio-key|--threedai-key)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      threedai_key="$2"
      shift 2
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

load_dotenv

docker_env_args=()
if [[ -n "$rodin_key" ]]; then
  docker_env_args+=(-e "HYPER3D_API_KEY=${rodin_key}")
elif [[ -n "${HYPER3D_API_KEY:-}" ]]; then
  docker_env_args+=(-e HYPER3D_API_KEY)
elif [[ -n "${RODIN_API_KEY:-}" ]]; then
  docker_env_args+=(-e "HYPER3D_API_KEY=${RODIN_API_KEY}")
fi

if [[ -n "$threedai_key" ]]; then
  docker_env_args+=(-e "THREEDAI_API_KEY=${threedai_key}")
elif [[ -n "${THREEDAI_API_KEY:-}" ]]; then
  docker_env_args+=(-e THREEDAI_API_KEY)
elif [[ -n "${THREEDAISTUDIO_API_KEY:-}" ]]; then
  docker_env_args+=(-e "THREEDAI_API_KEY=${THREEDAISTUDIO_API_KEY}")
elif [[ -n "${AI3DSTUDIO_API_KEY:-}" ]]; then
  docker_env_args+=(-e "THREEDAI_API_KEY=${AI3DSTUDIO_API_KEY}")
elif [[ -n "${THREEDAI_STUDIO_API_KEY:-}" ]]; then
  docker_env_args+=(-e "THREEDAI_API_KEY=${THREEDAI_STUDIO_API_KEY}")
elif threedai_key_from_dotenv="$(dotenv_value "3DAISTUDIO_API_KEY")" && [[ -n "$threedai_key_from_dotenv" ]]; then
  docker_env_args+=(-e "THREEDAI_API_KEY=${threedai_key_from_dotenv}")
fi

for env_name in \
  RODIN_API_BASE_URL RODIN_REQUEST_TIMEOUT RODIN_DOWNLOAD_TIMEOUT \
  HITEM3D_CLIENT_ID HITEM3D_CLIENT_SECRET HITEM3D_API_BASE_URL \
  HITEM3D_REQUEST_TIMEOUT HITEM3D_DOWNLOAD_TIMEOUT HITEM3D_POLL_INTERVAL HITEM3D_GENERATION_TIMEOUT \
  THREEDAI_API_BASE_URL THREEDAI_REQUEST_TIMEOUT THREEDAI_DOWNLOAD_TIMEOUT \
  THREEDAI_POLL_INTERVAL THREEDAI_GENERATION_TIMEOUT; do
  if [[ -n "${!env_name:-}" ]]; then
    docker_env_args+=(-e "$env_name")
  fi
done

exec docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "${port}:${port}" \
  -v "${repo_root}:/opt/TRELLIS.2" \
  -v "${hf_cache}:/root/.cache/huggingface" \
  -w /opt/TRELLIS.2 \
  "${docker_env_args[@]}" \
  "${image}" \
  python app.py --server-name 0.0.0.0 --server-port "${port}" "${share_args[@]}"
