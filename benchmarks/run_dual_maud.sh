#!/usr/bin/env bash
#
# Overnight MAUD embedding + Phase 2' analysis for the dual-embedding
# experiment (experiments/dual-embedding-plan.md §12, decision E).
#
#   bash benchmarks/run_dual_maud.sh [MAX_CONTRACTS]
#     MAX_CONTRACTS   default 50 (embedding wall-clock guard; the cache makes
#                     growing the sample later incremental)
#
# Pool is the culled 4-model set (decision F: nomic dropped). openai first
# (API-bound, validates the pipeline in minutes); locals smallest-first so
# early arms land even if the night runs short. Between arms ALL embed models
# are evicted (sticky-load fix from run_hier_ollama.sh). Embedding is
# resumable per batch. The final p2 pass needs every arm cached, hence
# --allow-embed is NOT passed: if an arm failed, p2 refuses instead of
# silently re-embedding on the spot.
set -u

PY=./.venv/Scripts/python.exe
MAXC="${1:-50}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="benchmarks/results/dual_maud_${STAMP}.log"
mkdir -p benchmarks/results

ARMS=("openai" "egemma" "arctic" "qwen3")
ALL_MODELS=(
  "nomic-embed-text" "embeddinggemma:300m" "bge-m3"
  "snowflake-arctic-embed2" "qwen3-embedding:0.6b"
)

log() { echo "$@" | tee -a "$LOG"; }

unload_all() {
  for m in "${ALL_MODELS[@]}"; do ollama stop "$m" >/dev/null 2>&1; done
  for _ in $(seq 1 60); do
    ollama ps 2>/dev/null | grep -qE "nomic-embed-text|embeddinggemma|bge-m3|arctic|qwen3" || return 0
    sleep 1
  done
}

log "=== dual-embedding MAUD leg | max_contracts=$MAXC | $STAMP ==="
declare -a STATUS=()
for key in "${ARMS[@]}"; do
  unload_all
  log ""
  log "########## EMBED ARM: $key ##########"
  "$PY" benchmarks/eval_dual.py embed --dataset maud --max-papers "$MAXC" --model-key "$key" 2>&1 | tee -a "$LOG"
  STATUS+=("$key -> exit ${PIPESTATUS[0]}")
done

unload_all
log ""
log "########## P2 CROSS-MODEL FUSION ##########"
"$PY" benchmarks/eval_dual.py p2 --dataset maud --max-papers "$MAXC" --tag phase2 2>&1 | tee -a "$LOG"
STATUS+=("p2 -> exit ${PIPESTATUS[0]}")

log ""
log "=== SUMMARY ==="
for s in "${STATUS[@]}"; do log "  $s"; done
log "Result JSON: benchmarks/results/dual_phase2_maud_all_*.json"
log "Full log: $LOG"
