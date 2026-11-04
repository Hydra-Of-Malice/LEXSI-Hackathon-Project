#!/usr/bin/env bash
# Download Hub files over plain HTTP with resume, for connections that drop. `hf download` can lose a whole file when a
# connection breaks partway. Keeps retrying until each file is complete.
#   bash scripts/fetch_resumable.sh OUT_DIR REPO/FILE [REPO/FILE ...]
#   e.g. bash scripts/fetch_resumable.sh cache/gguf Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
set -u
out="$1"; shift
mkdir -p "$out"
for item in "$@"; do
  repo="${item%/*}"; file="${item##*/}"
  url="https://huggingface.co/$repo/resolve/main/$file"
  auth=()
  [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  size=$(curl -sIL "${auth[@]}" "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
  until [ -f "$out/$file" ]; do
    curl -sSL -C - --connect-timeout 30 --speed-limit 1024 --speed-time 120 "${auth[@]}" \
      -o "$out/$file.part" "$url"
    have=$(stat -c %s "$out/$file.part" 2>/dev/null || echo 0)
    if [ -n "$size" ] && [ "$have" -ge "$size" ]; then
      mv "$out/$file.part" "$out/$file"
      echo "done $file ($have bytes)"
    else
      echo "retrying $file at $have of ${size:-unknown} bytes" >&2
      sleep 10
    fi
  done
done
