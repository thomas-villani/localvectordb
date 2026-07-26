#!/usr/bin/env bash
#
# Overnight Phase 0 run for the dual-embedding experiment
# (experiments/dual-embedding-plan.md §5).
#
#   bash benchmarks/run_dual_phase0.sh [SPLIT]
#     SPLIT   default "dev" (all of Qasper dev, per decision D)
#
# Embeds chunks/sections/queries for every pool model into the shared disk
# cache (benchmarks/.cache/hier_embed/), one model at a time, then runs the
# P0-A/B/C analysis over whichever arms completed. Embedding is resumable: the
# cache is written per batch, so a crashed/interrupted arm continues where it
# left off on the next invocation.
#
# openai runs first: it is API-bound (fast), validates the full pipeline early,
# and reuses the original hierarchical-study cache where texts overlap. Local
# arms run smallest-model-first so early results land even if the night runs
# short. Between arms ALL embed models are evicted (little free RAM on this
# box, and eviction forces a fresh load at the arm's num_ctx/num_batch --
# the sticky-load fix from run_hier_ollama.sh).
set -u

PY=./.venv/Scripts/python.exe
SPLIT="${1:-dev}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="benchmarks/results/dual_phase0_${STAMP}.log"
mkdir -p benchmarks/results

ARMS=("openai" "nomic" "egemma" "arctic" "qwen3")
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

log "=== dual-embedding Phase 0 | split=$SPLIT | $STAMP ==="
declare -a STATUS=()
COMPLETED=""
for key in "${ARMS[@]}"; do
  unload_all
  log ""
  log "########## EMBED ARM: $key ##########"
  "$PY" benchmarks/eval_dual.py embed --model-key "$key" --split "$SPLIT" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  STATUS+=("$key -> exit $rc")
  if [ "$rc" -eq 0 ]; then
    COMPLETED="${COMPLETED:+$COMPLETED,}$key"
  fi
done

unload_all
log ""
log "########## ANALYZE: $COMPLETED ##########"
"$PY" benchmarks/eval_dual.py analyze --models "$COMPLETED" --split "$SPLIT" 2>&1 | tee -a "$LOG"
STATUS+=("analyze($COMPLETED) -> exit ${PIPESTATUS[0]}")

log ""
log "=== SUMMARY ==="
for s in "${STATUS[@]}"; do log "  $s"; done
log "Result JSON: benchmarks/results/dual_phase0_qasper_${SPLIT}_*.json"
log "Full log: $LOG"
