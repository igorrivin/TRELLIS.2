#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/share-glb-gcs.sh GLB_PATH [--bucket gs://bucket] [--prefix path] [--public] [--duration 7d]

Uploads a GLB to Google Cloud Storage and prints either a signed URL or, with
--public, a public storage URL.

Environment:
  GLB_SHARE_BUCKET   Default bucket. Defaults to gs://heretic-batch-audit.
  GLB_SHARE_PREFIX   Default object prefix. Defaults to trellis-glb/<timestamp>-<stem>.
  GCS_SIGN_DURATION  Signed URL duration. Defaults to 7d.
USAGE
}

glb_path=""
bucket="${GLB_SHARE_BUCKET:-gs://heretic-batch-audit}"
prefix=""
public=0
duration="${GCS_SIGN_DURATION:-7d}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)
      bucket="$2"
      shift 2
      ;;
    --prefix)
      prefix="$2"
      shift 2
      ;;
    --public)
      public=1
      shift
      ;;
    --duration)
      duration="$2"
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
      if [[ -n "$glb_path" ]]; then
        echo "Unexpected extra argument: $1" >&2
        usage >&2
        exit 2
      fi
      glb_path="$1"
      shift
      ;;
  esac
done

if [[ -z "$glb_path" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$glb_path" ]]; then
  echo "GLB file does not exist: $glb_path" >&2
  exit 1
fi
if [[ "${glb_path,,}" != *.glb ]]; then
  echo "Expected a .glb file: $glb_path" >&2
  exit 1
fi
if [[ "$bucket" != gs://* ]]; then
  echo "Bucket must be a gs:// URL." >&2
  exit 1
fi

stem="$(basename "$glb_path" .glb)"
if [[ -z "$prefix" ]]; then
  prefix="trellis-glb/$(date -u +%Y%m%dT%H%M%SZ)-${stem}"
fi
prefix="${prefix#/}"
prefix="${prefix%/}"
bucket="${bucket%/}"
object="${bucket}/${prefix}/$(basename "$glb_path")"

gsutil -h "Content-Type:model/gltf-binary" cp "$glb_path" "$object" >/dev/null

bucket_name="${bucket#gs://}"
encoded_object_path="${prefix}/$(basename "$glb_path")"
public_url="https://storage.googleapis.com/${bucket_name}/${encoded_object_path}"

echo "Uploaded: $object"
if (( public )); then
  gsutil acl ch -u AllUsers:R "$object" >/dev/null
  echo "$public_url"
  exit 0
fi

if signed_url="$(gcloud storage sign-url "$object" --duration="$duration" --format='value(signed_url)' 2>/dev/null)"; then
  echo "$signed_url"
else
  echo "Could not create signed URL automatically. GCS object path:" >&2
  echo "$object"
fi
