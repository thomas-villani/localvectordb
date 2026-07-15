#!/usr/bin/env bash
#
# Overnight hierarchical-retrieval experiment across local Ollama encoders and
# chunk sizes.
#
#   bash benchmarks/run_hier_ollama.sh [MAX_PAPERS] [CHUNK_SIZES]
#     MAX_PAPERS   default 40
#     CHUNK_SIZES  space-separated, default "500 1000"
#
# Runs benchmarks/eval_hierarchical.py once per (encoder arm x chunk size) on
# Qasper (section mode). Each run's raw vectors and JSON result are cached/written
# by the harness (embed cache keyed per model+num_ctx; results per model+ctx+chunk).
#
# Two axes:
#   * encoder / context : nomic@2k, embeddinggemma@2k, bge-m3@2k, bge-m3@8k
#   * chunk size        : 500 (shipped default) vs 1000 (large-chunk baseline)
# On Qasper (~4.3 chars/token) a 1000-token chunk maxes ~5.5k chars, under the
# ~6144-char single-pass window on a 2k model, so a large chunk is embedded in ONE
# pass (not window-pooled) on every arm -- a faithful large-chunk baseline. (>=1200
# tokens starts pooling on 2k models; >=1750 pools ~90%.) The chunk-size axis
# answers "do the section/fusion arms still beat a LARGE-chunk baseline, or is the
# hierarchical win just an artefact of small chunks?" -- compare `chunk` against
# `rawspan-section`/`fusion-rawspan` at each chunk size.
#
# Arms (the 8k arm needs the num_batch fix -- Ollama's /api/embed caps input at
# n_batch=2048 regardless of num_ctx; OllamaEmbeddings auto-sets num_batch=num_ctx):
#   nomic-embed-text     num_ctx=2048   (arch-capped at 2048)
#   embeddinggemma:300m  num_ctx=2048   (arch-capped at 2048)
#   bge-m3               num_ctx=2048   (2k baseline for the context axis)
#   bge-m3               num_ctx=8192   (true 8k -- only reachable via num_batch)
#
# Between arms the model is fully unloaded (polled via `ollama ps`) so Ollama
# reloads it fresh at this arm's num_ctx/num_batch -- otherwise a "2k" arm can
# silently reuse an 8k-loaded model (sticky load).
set -u

PY=./.venv/Scripts/python.exe
PAPERS="${1:-40}"
CHUNKS="${2:-500 1000}"
SPLIT="dev"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="benchmarks/results/hier_ollama_${STAMP}.log"
mkdir -p benchmarks/results

# "model num_ctx timeout" per arm. TIMEOUT is the per-request read timeout in
# seconds. bge-m3 is a 1.2GB / 1024-dim model and on this CPU-only box a single
# batch takes >300s (the provider default) at ANY context -- the 15-paper run
# proved bge-m3@2048 flakily times out at 300s, not just the 8k arm -- so every
# bge-m3 arm gets 1800s. nomic/embeddinggemma (smaller) finish inside 300s.
ARMS=(
  "nomic-embed-text 2048 300"
  "embeddinggemma:300m 2048 300"
  "bge-m3 2048 1800"
  "bge-m3 8192 1800"
)

log() { echo "$@" | tee -a "$LOG"; }

# All embed models this sweep touches. Before each arm we evict ALL of them so
# only the current arm's model is resident -- this box has little free RAM and a
# second 1.2GB model resident makes Ollama thrash. Evicting also forces a fresh
# load at the arm's num_ctx/num_batch (the sticky-load fix: a "2k" arm must not
# reuse an 8k-loaded model). Waits up to ~60s for them to leave `ollama ps`.
ALL_MODELS=("nomic-embed-text" "embeddinggemma:300m" "bge-m3")
unload_all() {
  for m in "${ALL_MODELS[@]}"; do ollama stop "$m" >/dev/null 2>&1; done
  for _ in $(seq 1 60); do
    ollama ps 2>/dev/null | grep -qE "nomic-embed-text|embeddinggemma|bge-m3" || return 0
    sleep 1
  done
}

log "=== hierarchical Ollama sweep | papers=$PAPERS split=$SPLIT chunks='$CHUNKS' | $STAMP ==="
declare -a STATUS=()
for arm in "${ARMS[@]}"; do
  set -- $arm; MODEL="$1"; CTX="$2"; TIMEOUT="$3"
  # Evict everything once per arm; the model then loads fresh at this arm's
  # num_ctx/num_batch and stays warm across the chunk sizes (chunk size doesn't
  # change the loaded model, so no reload is needed between them).
  unload_all
  for CK in $CHUNKS; do
    log ""
    log "########## ARM: $MODEL  num_ctx=$CTX  chunk_tokens=$CK  timeout=${TIMEOUT}s ##########"
    "$PY" benchmarks/eval_hierarchical.py --dataset qasper --split "$SPLIT" \
      --max-papers "$PAPERS" --mode section \
      --provider ollama --model "$MODEL" --num-ctx "$CTX" --chunk-tokens "$CK" \
      --timeout "$TIMEOUT" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    STATUS+=("$MODEL@${CTX} ck=$CK -> exit $rc")
  done
done

log ""
log "=== SUMMARY ==="
for s in "${STATUS[@]}"; do log "  $s"; done
log ""
log "Per-arm JSON results: benchmarks/results/hierarchical_qasper_${SPLIT}_*.json"
log "Full log: $LOG"
