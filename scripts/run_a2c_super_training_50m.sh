#!/usr/bin/env bash
# Launcher operativo per lo scouting A2C v14 da 50M, diviso in blocchi da 10M.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_REL="benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723"
RUN_DIR="$ROOT_DIR/$RUN_REL"
PREFIX="a2c_v14_scale50m_seed20260723"
PID_FILE="$RUN_DIR/train.pid"
LOG_POINTER="$RUN_DIR/current_log"
COMMIT_FILE="$RUN_DIR/run.commit"

usage() {
  cat <<'EOF'
Uso:
  scripts/run_a2c_super_training_50m.sh start 10   # avvia soltanto il blocco 0-10M
  scripts/run_a2c_super_training_50m.sh status     # mostra processo e ultimo log
  scripts/run_a2c_super_training_50m.sh log        # segue il log corrente

Dopo lo screen a 10M, lo stesso launcher accetta esplicitamente start 20, 30, 40 o 50.
Non concatena mai i blocchi automaticamente.
EOF
}

read_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  tr -d '[:space:]' < "$PID_FILE"
}

is_running() {
  local pid
  pid="$(read_pid)" || return 1
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local process_command
  process_command="$(ps -p "$pid" -o command= 2>/dev/null)" || return 1
  [[ "$process_command" == *"scripts/train_a2c.py"* ]]
}

show_status() {
  if is_running; then
    local pid
    pid="$(read_pid)"
    echo "Training attivo:"
    ps -p "$pid" -o pid=,etime=,%cpu=,rss=,command=
  else
    echo "Nessun processo di training attivo."
  fi

  if [[ -f "$LOG_POINTER" ]]; then
    local log_path
    log_path="$(<"$LOG_POINTER")"
    echo
    echo "Ultime righe di $log_path:"
    tail -n 12 "$log_path" 2>/dev/null || true
  fi
}

start_block() {
  local target_m="${1:-}"
  case "$target_m" in
    10|20|30|40|50) ;;
    *)
      echo "Target non valido: usare 10, 20, 30, 40 o 50." >&2
      exit 2
      ;;
  esac

  if is_running; then
    echo "Esiste già un training attivo (PID $(read_pid)). Usa 'status'." >&2
    exit 1
  fi

  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Il repository contiene modifiche tracked non committate: congelarle prima del training." >&2
    exit 1
  fi

  command -v uv >/dev/null || { echo "uv non trovato." >&2; exit 1; }
  command -v caffeinate >/dev/null || { echo "caffeinate non trovato." >&2; exit 1; }

  mkdir -p "$RUN_DIR/models" "$RUN_DIR/resume" "$RUN_DIR/validation" "$RUN_DIR/final_gate"

  local current_commit
  current_commit="$(git rev-parse HEAD)"
  if [[ "$target_m" == "10" ]]; then
    if [[ -f "$COMMIT_FILE" ]] && [[ "$(<"$COMMIT_FILE")" != "$current_commit" ]]; then
      echo "La directory appartiene a un altro commit; non sovrascrivo la provenienza." >&2
      exit 1
    fi
    printf '%s\n' "$current_commit" > "$COMMIT_FILE"
  else
    if [[ ! -f "$COMMIT_FILE" ]] || [[ "$(<"$COMMIT_FILE")" != "$current_commit" ]]; then
      echo "Il commit corrente non coincide con quello che ha iniziato il run." >&2
      exit 1
    fi
  fi

  local start_m=$((target_m - 10))
  local start_games=$((start_m * 1000000))
  local midpoint_games=$((start_games + 5000000))
  local target_games=$((target_m * 1000000))
  local output="$RUN_DIR/models/${PREFIX}_at${target_m}m.npz"
  local midpoint_checkpoint="$RUN_DIR/resume/${PREFIX}_$((start_m + 5))m.npz"
  local target_checkpoint="$RUN_DIR/resume/${PREFIX}_${target_m}m.npz"
  local diagnostics="$RUN_DIR/diagnostics_0_${target_m}m.sampled.json"
  local log_path="$RUN_DIR/train_${start_m}_${target_m}m.log"

  if [[ -e "$output" || -e "$midpoint_checkpoint" || -e "$target_checkpoint" ]]; then
    echo "Il blocco ha già artefatti di output; non li sovrascrivo:" >&2
    printf '  %s\n' "$output" "$midpoint_checkpoint" "$target_checkpoint" >&2
    exit 1
  fi

  local start_args
  if [[ "$target_m" == "10" ]]; then
    start_args=(--init data/models/best_a2c_v14.npz)
  else
    local resume_checkpoint="$RUN_DIR/resume/${PREFIX}_${start_m}m.npz"
    if [[ ! -f "$resume_checkpoint" ]]; then
      echo "Checkpoint di partenza mancante: $resume_checkpoint" >&2
      exit 1
    fi
    start_args=(--resume "$resume_checkpoint")
  fi

  local command=(
    uv run python scripts/train_a2c.py
    "${start_args[@]}"
    --out "$output"
    --encoder-version v4
    --rollout-engine fast --fast-rollout numba
    --training-schedule serial --seat-fair
    --opponent-mix "bc_model:0.15,bc_model_pimc_belief:0.40,bc_model_value_lookahead_8x8:0.20,heuristic_trump_saver:0.12,heuristic_v1:0.04,heuristic_v2:0.06,random:0.03"
    --opponent-model data/models/best_a2c_v14.npz
    --opponent-belief-model data/models/belief_v0_h128_50k_seed20260702.npz
    --opponent-pimc-determinizations 16
    --opponent-value-model data/models/value_v1_v4_fullgame_h128_seed20260718.npz
    --opponent-value-max-unknown-cards 8
    --bc-anchor data/models/best_a2c_v14.npz --bc-anchor-beta 0.01
    --overkill-penalty-mode gap --overkill-penalty-beta 0.3
    --overkill-low-lead-points-max 2
    --lr 0.0003 --weight-decay 0
    --entropy-beta 0.0005 --value-coef 0.5 --gamma 1.0
    --suit-augmentation off --suit-consistency-beta 0 --suit-margin-beta 0
    --update-every 20 --log-every 1000
    --metrics-mode summary
    --diagnostics-json "$diagnostics" --diagnostics-every 1000
    --num-games 50000000 --stop-after-games "$target_games"
    --checkpoint-games "$midpoint_games,$target_games"
    --checkpoint-dir "$RUN_DIR/resume"
    --checkpoint-prefix "$PREFIX"
    --seed 20260723
  )

  printf '%s\n' "$log_path" > "$LOG_POINTER"
  nohup env PYTHONUNBUFFERED=1 caffeinate -dimsu "${command[@]}" > "$log_path" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"

  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Il processo è terminato durante l'avvio. Ultime righe:" >&2
    tail -n 30 "$log_path" >&2 || true
    exit 1
  fi

  echo "Avviato blocco ${start_m}-${target_m}M (PID $pid)."
  echo "Log: $log_path"
  echo "Stato: scripts/run_a2c_super_training_50m.sh status"
  echo "Log live: scripts/run_a2c_super_training_50m.sh log"
}

case "${1:-}" in
  start)
    start_block "${2:-}"
    ;;
  status)
    show_status
    ;;
  log)
    if [[ ! -f "$LOG_POINTER" ]]; then
      echo "Nessun log registrato per questo esperimento." >&2
      exit 1
    fi
    tail -f "$(<"$LOG_POINTER")"
    ;;
  help|-h|--help|"")
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
