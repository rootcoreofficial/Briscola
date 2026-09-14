#!/usr/bin/env bash
# Launcher congelato per distillare il teacher 20M a 24 viste su 250k partite.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_REL="benchmarks/experiments/suit_distillation_20m_teacher24_250k_seed20260724"
RUN_DIR="$ROOT_DIR/$RUN_REL"
TEACHER_MODEL_REL="benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/models/a2c_v14_scale50m_seed20260723_at20m.npz"
TEACHER_MODEL="$ROOT_DIR/$TEACHER_MODEL_REL"
DATA_DIR="$RUN_DIR/dataset"
MANIFEST="$DATA_DIR/manifest.json"
MODEL_OUT="$RUN_DIR/models/suit_distilled_20m_teacher24_250k_seed20260724.npz"
TRAIN_REPORT="$RUN_DIR/training_report.json"
SYMMETRY_REPORT="$RUN_DIR/suit_symmetry.json"
SCREEN_V14_REPORT="$RUN_DIR/screen_student_vs_v14_20k.json"
CONFIRM_V14_REPORT="$RUN_DIR/confirm_student_vs_v14_100k.json"
QUALITY_REPORT="$RUN_DIR/decision_quality_student_vs_heuristic_v1_medium.json"
PIMC_REPORT="$RUN_DIR/pimc16x8_student_vs_v14_10k.json"
PIMC_LOG="$RUN_DIR/pimc16x8_student_vs_v14_10k.log"
EFFICIENCY_12_SCREEN="$RUN_DIR/efficiency_12x8/screen_student12_vs_v14_16_4k.json"
EFFICIENCY_12_CONFIRM="$RUN_DIR/efficiency_12x8/confirm_student12_vs_v14_16_20k.json"
EFFICIENCY_12_LOG="$RUN_DIR/efficiency_12x8/confirm_student12_vs_v14_16_20k.log"
VERIFY_RECEIPT="$RUN_DIR/dataset_verified.sha256"
PID_FILE="$RUN_DIR/current.pid"
LOG_POINTER="$RUN_DIR/current_log"
STAGE_POINTER="$RUN_DIR/current_stage"

usage() {
  cat <<'EOF'
Uso:
  scripts/run_suit_distillation_20m_250k.sh start-data   # genera/riprende 10 shard da 25k
  scripts/run_suit_distillation_20m_250k.sh status       # processo e ultime righe del log
  scripts/run_suit_distillation_20m_250k.sh log          # segue il log corrente
  scripts/run_suit_distillation_20m_250k.sh verify-data  # verifica hash e contenuto dei 10 shard
  scripts/run_suit_distillation_20m_250k.sh start-train  # solo dopo il gate manuale sul manifest
  scripts/run_suit_distillation_20m_250k.sh start-pimc   # gate finale live, dopo i report policy/qualita'
  scripts/run_suit_distillation_20m_250k.sh start-12x8-confirm  # conferma efficienza dopo lo screen GO

La raccolta e il training non vengono concatenati: il manifest completo deve essere
controllato prima di autorizzare lo student.
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
  [[ "$process_command" == *"suit_distillation"* ]]
}

require_runtime() {
  command -v uv >/dev/null || { echo "uv non trovato." >&2; exit 1; }
  command -v caffeinate >/dev/null || { echo "caffeinate non trovato." >&2; exit 1; }
  [[ -f "$TEACHER_MODEL" ]] || { echo "Teacher 20M mancante: $TEACHER_MODEL_REL" >&2; exit 1; }
  mkdir -p "$RUN_DIR/models"
}

show_status() {
  if is_running; then
    local pid
    pid="$(read_pid)"
    local stage="sconosciuto"
    [[ -f "$STAGE_POINTER" ]] && stage="$(<"$STAGE_POINTER")"
    echo "Job attivo, fase $stage:"
    ps -p "$pid" -o pid=,etime=,%cpu=,rss=,command=
  else
    echo "Nessun job di distillazione attivo."
  fi

  if [[ -f "$LOG_POINTER" ]]; then
    local log_path
    log_path="$(<"$LOG_POINTER")"
    echo
    echo "Ultime righe di $log_path:"
    tail -n 16 "$log_path" 2>/dev/null || true
  fi
}

start_background() {
  local stage="$1"
  local log_path="$2"
  shift 2
  local command=("$@")

  if is_running; then
    echo "Esiste già un job attivo (PID $(read_pid)). Usa 'status'." >&2
    exit 1
  fi
  printf '%s\n' "$stage" > "$STAGE_POINTER"
  printf '%s\n' "$log_path" > "$LOG_POINTER"
  nohup env PYTHONUNBUFFERED=1 caffeinate -dimsu "${command[@]}" >> "$log_path" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"

  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Il processo è terminato durante l'avvio. Ultime righe:" >&2
    tail -n 30 "$log_path" >&2 || true
    exit 1
  fi
  echo "Avviata fase $stage (PID $pid)."
  echo "Log: $log_path"
  echo "Stato: scripts/run_suit_distillation_20m_250k.sh status"
  echo "Log live: scripts/run_suit_distillation_20m_250k.sh log"
}

start_data() {
  require_runtime
  if [[ -f "$MANIFEST" ]]; then
    local manifest_status
    manifest_status="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$MANIFEST")"
    if [[ "$manifest_status" == "complete" ]]; then
      echo "Il corpus è già completo. Eseguire 'verify-data'."
      return
    fi
  fi
  local log_path="$RUN_DIR/data_generation.log"
  local command=(
    uv run python scripts/generate_suit_distillation_shards.py
    --model "$TEACHER_MODEL"
    --out-dir "$DATA_DIR"
    --num-games 250000
    --games-per-shard 25000
    --seed 20260724
    --opponent-mix "mirror:0.50,heuristic_trump_saver:0.20,heuristic_v1:0.15,heuristic_v2:0.10,random:0.05"
    --temperature 1.0
    --train-fraction 0.8
    --validation-fraction 0.1
    --progress-every 1000
    --resume
  )
  start_background "data" "$log_path" "${command[@]}"
}

verify_data() {
  require_runtime
  if is_running; then
    echo "La raccolta è ancora attiva; attendere prima della verifica." >&2
    exit 1
  fi
  [[ -f "$MANIFEST" ]] || { echo "Manifest mancante: $MANIFEST" >&2; exit 1; }
  uv run python scripts/generate_suit_distillation_shards.py \
    --model "$TEACHER_MODEL" \
    --out-dir "$DATA_DIR" \
    --num-games 250000 \
    --games-per-shard 25000 \
    --seed 20260724 \
    --opponent-mix "mirror:0.50,heuristic_trump_saver:0.20,heuristic_v1:0.15,heuristic_v2:0.10,random:0.05" \
    --temperature 1.0 \
    --train-fraction 0.8 \
    --validation-fraction 0.1 \
    --progress-every 0 \
    --verify-only
  local manifest_sha256
  manifest_sha256="$(uv run python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$MANIFEST")"
  printf '%s\n' "$manifest_sha256" > "$VERIFY_RECEIPT"
  echo "Ricevuta verifica: $VERIFY_RECEIPT"
}

start_train() {
  require_runtime
  if is_running; then
    echo "Esiste già un job attivo (PID $(read_pid)). Usa 'status'." >&2
    exit 1
  fi
  [[ -f "$MANIFEST" ]] || { echo "Manifest mancante: $MANIFEST" >&2; exit 1; }
  [[ -f "$VERIFY_RECEIPT" ]] || {
    echo "Ricevuta di verifica mancante: eseguire prima 'verify-data'." >&2
    exit 1
  }
  [[ ! -e "$MODEL_OUT" && ! -e "$TRAIN_REPORT" ]] || {
    echo "Output training già presente; non lo sovrascrivo." >&2
    exit 1
  }
  local manifest_status
  manifest_status="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$MANIFEST")"
  [[ "$manifest_status" == "complete" ]] || {
    echo "Il manifest non è completo: stato $manifest_status" >&2
    exit 1
  }
  local verified_sha256 current_sha256
  verified_sha256="$(<"$VERIFY_RECEIPT")"
  current_sha256="$(uv run python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$MANIFEST")"
  [[ "$verified_sha256" == "$current_sha256" ]] || {
    echo "Il manifest è cambiato dopo verify-data; ripetere la verifica." >&2
    exit 1
  }
  local log_path="$RUN_DIR/training.log"
  local command=(
    uv run python scripts/train_suit_distillation_shards.py
    --manifest "$MANIFEST"
    --init "$TEACHER_MODEL"
    --out "$MODEL_OUT"
    --report-json "$TRAIN_REPORT"
    --epochs 5
    --batch-size 1024
    --lr 0.0002
    --weight-decay 0.000001
    --seed 20260724
    --label "Distillazione teacher 20M 250k"
  )
  start_background "train" "$log_path" "${command[@]}"
}

start_pimc() {
  require_runtime
  local required_report
  for required_report in \
    "$MODEL_OUT" \
    "$TRAIN_REPORT" \
    "$SYMMETRY_REPORT" \
    "$SCREEN_V14_REPORT" \
    "$CONFIRM_V14_REPORT" \
    "$QUALITY_REPORT"; do
    [[ -f "$required_report" ]] || {
      echo "Gate precedente mancante: $required_report" >&2
      exit 1
    }
  done
  [[ -f "$ROOT_DIR/data/models/belief_v0_h128_50k_seed20260702.npz" ]] || {
    echo "Belief v0 ufficiale mancante in data/models/." >&2
    exit 1
  }
  [[ ! -e "$PIMC_REPORT" ]] || {
    echo "Report PIMC gia' presente; non lo sovrascrivo: $PIMC_REPORT" >&2
    exit 1
  }

  local command=(
    uv run python scripts/evaluate_agents.py
    --engine domain
    --num-games 10000
    --seed 0
    --seat-fair
    --seed-suite-range-start 12000000
    --agent0 bc_model_pimc_belief_16x8
    --agent0-model "$MODEL_OUT"
    --agent1 bc_model_pimc_belief_16x8
    --agent1-model "$ROOT_DIR/data/models/best_a2c_v14.npz"
    --out-json "$PIMC_REPORT"
  )
  start_background "pimc" "$PIMC_LOG" "${command[@]}"
}

start_12x8_confirm() {
  require_runtime
  [[ -f "$MODEL_OUT" ]] || {
    echo "Student 250k mancante: $MODEL_OUT" >&2
    exit 1
  }
  [[ -f "$EFFICIENCY_12_SCREEN" ]] || {
    echo "Screen 12x8 mancante: $EFFICIENCY_12_SCREEN" >&2
    exit 1
  }
  [[ -f "$ROOT_DIR/data/models/best_a2c_v14.npz" ]] || {
    echo "Policy v14 ufficiale mancante in data/models/." >&2
    exit 1
  }
  [[ -f "$ROOT_DIR/data/models/belief_v0_h128_50k_seed20260702.npz" ]] || {
    echo "Belief v0 ufficiale mancante in data/models/." >&2
    exit 1
  }
  [[ ! -e "$EFFICIENCY_12_CONFIRM" ]] || {
    echo "Report 12x8 gia' presente; non lo sovrascrivo: $EFFICIENCY_12_CONFIRM" >&2
    exit 1
  }

  mkdir -p "$(dirname "$EFFICIENCY_12_CONFIRM")"
  local command=(
    uv run python scripts/evaluate_pimc.py
    --model "$MODEL_OUT"
    --num-games 20000
    --seed 20260728
    --determinizations 12
    --max-unknown-cards 8
    --opponent pimc
    --opponent-model "$ROOT_DIR/data/models/best_a2c_v14.npz"
    --opponent-determinizations 16
    --opponent-max-unknown-cards 8
    --belief-model "$ROOT_DIR/data/models/belief_v0_h128_50k_seed20260702.npz"
    --opponent-belief-model "$ROOT_DIR/data/models/belief_v0_h128_50k_seed20260702.npz"
    --belief-uniform-mix 0.10
    --opponent-belief-uniform-mix 0.10
    --out-json "$EFFICIENCY_12_CONFIRM"
  )
  start_background "12x8-confirm" "$EFFICIENCY_12_LOG" "${command[@]}"
}

case "${1:-}" in
  start-data)
    start_data
    ;;
  verify-data)
    verify_data
    ;;
  start-train)
    start_train
    ;;
  start-pimc)
    start_pimc
    ;;
  start-12x8-confirm)
    start_12x8_confirm
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
