#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# run_v8_tuned.sh
#
# Script prêt-à-l'emploi pour lancer simulation_V8_23_rich_env_tuned.py
# - venv optionnel
# - installation dépendances minimales (numpy, scikit-learn, matplotlib)
# - lancement en tmux (recommandé) ou en nohup
# - rotation simple des logs
# - conversion portable top5(.npz) -> JSON (option)
#
# Usage examples:
#   ./run_v8_tuned.sh --tuned             # run complet (script tuned)
#   ./run_v8_tuned.sh --short             # run court pour test (tuned_short)
#   ./run_v8_tuned.sh --tuned --tmux      # run dans tmux session "simrun"
#   ./run_v8_tuned.sh --convert-top5 F    # convert F (ex: top5_v8_23_gen_080.npz) -> JSON
#
# Variables d'environnement utiles (optionnel) :
#   NUMPY_VERSION (ex: 1.24.4) - si tu veux forcer une version numpy compatible pickle
#   PYTHON (ex: /usr/bin/python3) - chemin vers python
#
set -euo pipefail
IFS=$'\n\t'

# -------------------------
# Configuration par défaut
# -------------------------
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv_sim_v8}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
TUNED_SCRIPT="${TUNED_SCRIPT:-simulation_V8_23_rich_env_tuned.py}"
SHORT_SCRIPT="${SHORT_SCRIPT:-simulation_V8_23_rich_env_tuned_short.py}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
TMUX_SESSION="${TMUX_SESSION:-simrun}"
NUMPY_VERSION="${NUMPY_VERSION:-}"   # si vide, pip install numpy latest
PIP_ARGS="${PIP_ARGS:-}"            # optionnel, ex: --no-cache-dir

mkdir -p "$LOG_DIR"

# -------------------------
# Helpers
# -------------------------
timestamp() { date +"%Y%m%d_%H%M%S"; }

rotate_log() {
  # rotation simple : garde 7 fichiers
  local logfile="$1"
  if [ -f "$logfile" ]; then
    local ts
    ts=$(timestamp)
    mv "$logfile" "${logfile%.log}_$ts.log"
    # prune older files (garde 7)
    ls -1t "${logfile%.log}"*_*.log 2>/dev/null | tail -n +8 | xargs -r rm -f
  fi
}

ensure_python() {
  if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Erreur : python introuvable ($PYTHON). Installe Python 3 et réessaie."
    exit 1
  fi
}

create_venv_and_install() {
  echo "==> Création d'un venv ($VENV_DIR) et installation dépendances minimales"
  "$PYTHON" -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  if [ -n "$NUMPY_VERSION" ]; then
    pip install $PIP_ARGS "numpy==${NUMPY_VERSION}"
  else
    pip install $PIP_ARGS numpy
  fi
  pip install $PIP_ARGS scipy scikit-learn matplotlib
  echo "==> venv prêt. Pour activer manuellement : source $VENV_DIR/bin/activate"
}

run_in_tmux() {
  local script="$1"
  local logfile="$2"
  rotate_log "$logfile"
  echo "==> Lancement dans tmux session '${TMUX_SESSION}' : $script"
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux n'est pas installé. Installe tmux ou relance sans --tmux."
    exit 1
  fi
  # create session if not exists
  if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_DIR"
  fi
  # run the script in a new window, log with tee
  tmux new-window -t "${TMUX_SESSION}:" -n "run_$(timestamp)" -c "$PROJECT_DIR" \
    "bash -lc 'source \"$VENV_DIR/bin/activate\" 2>/dev/null || true; python3 \"$script\" 2>&1 | tee \"$logfile\"; echo \"PROCESS_DONE:\$(date +%s)\" >> \"$logfile\"'"
  echo "Attache-toi à la session : tmux attach -t $TMUX_SESSION"
  echo "Log: $logfile"
}

run_nohup() {
  local script="$1"
  local logfile="$2"
  rotate_log "$logfile"
  echo "==> Lancement en background (nohup) : $script"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate" 2>/dev/null || true
  nohup python3 "$script" > "$logfile" 2>&1 &
  local pid=$!
  echo "$pid" > "${logfile%.log}.pid"
  echo "PID: $pid  — log: $logfile"
}

run_foreground() {
  local script="$1"
  local logfile="$2"
  rotate_log "$logfile"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate" 2>/dev/null || true
  echo "==> Lancement en avant-plan (foreground) : $script"
  python3 "$script" 2>&1 | tee "$logfile"
}

convert_top5_to_json() {
  # convertit un fichier .npz contenant top5 picklés en JSON (tentative - dépend compatibilité numpy)
  local npzfile="$1"
  local outjson="${npzfile%.npz}_portable.json"
  echo "==> Conversion $npzfile -> $outjson"
  python3 - <<'PY'
import sys, json, numpy as np
from pathlib import Path

npz = sys.argv[1]
out = sys.argv[2]
data = np.load(npz, allow_pickle=True)
# try to find array/object named 'top5' or first key
k = None
for key in data.files:
    if key == 'top5':
        k = key; break
if k is None:
    k = data.files[0]
top5 = data[k]
simple = []
for idx, ind in enumerate(top5):
    d = {}
    if hasattr(ind, "items"):
        for kk, vv in ind.items():
            try:
                import numpy as _np
                if isinstance(vv, _np.ndarray):
                    d[kk] = vv.tolist()
                else:
                    # try json serializable
                    d[kk] = vv
            except Exception:
                d[kk] = str(vv)
    else:
        # fallback: try to convert numpy array
        try:
            d = {"value": ind.tolist()}
        except Exception:
            d = {"repr": repr(ind)}
    simple.append(d)
Path(out).write_text(json.dumps(simple, indent=2))
print("wrote", out)
PY
  if [ $? -eq 0 ]; then
    echo "Conversion terminée : $outjson"
  else
    echo "Erreur lors de la conversion."
  fi
}

# -------------------------
# Argument parsing
# -------------------------
MODE=""
LAUNCH_MODE="tmux"  # default: tmux; options: tmux|nohup|fg
while [ $# -gt 0 ]; do
  case "$1" in
    --tuned) MODE="tuned"; shift ;;
    --short) MODE="short"; shift ;;
    --tmux) LAUNCH_MODE="tmux"; shift ;;
    --nohup) LAUNCH_MODE="nohup"; shift ;;
    --fg|--foreground) LAUNCH_MODE="fg"; shift ;;
    --no-venv) NO_VENV=1; shift ;;
    --venv) NO_VENV=0; shift ;;
    --convert-top5) NPZ="$2"; shift 2 ;;
    --help|-h) echo "Usage: $0 [--tuned|--short] [--tmux|--nohup|--fg] [--convert-top5 FILE]"; exit 0 ;;
    *) echo "Option inconnue: $1"; echo "Usage: $0 [--tuned|--short] [--tmux|--nohup|--fg] [--convert-top5 FILE]"; exit 1 ;;
  esac
done

ensure_python

# si demande de conversion top5
if [ "${NPZ:-}" != "" ]; then
  if [ ! -f "$NPZ" ]; then echo "Fichier introuvable: $NPZ"; exit 1; fi
  convert_top5_to_json "$NPZ"
  exit 0
fi

if [ -z "${MODE}" ]; then
  echo "Aucune action sélectionnée. Passe --tuned ou --short ou --convert-top5 FILE"
  exit 1
fi

# vérifie que les scripts existent
target_script="$PROJECT_DIR/$TUNED_SCRIPT"
if [ "$MODE" = "short" ]; then
  target_script="$PROJECT_DIR/$SHORT_SCRIPT"
fi
if [ ! -f "$target_script" ]; then
  echo "Script cible introuvable: $target_script"
  exit 1
fi

# setup venv si nécessaire
if [ "${NO_VENV:-0}" != "1" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    create_venv_and_install
  else
    echo "Venv existant trouvé : $VENV_DIR (activation)"
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate" 2>/dev/null || true
  fi
fi

# prepare log name
LOGFILE="$LOG_DIR/run_$(basename "$target_script" .py)_$(timestamp).log"

# run according to chosen mode
case "$LAUNCH_MODE" in
  tmux)
    run_in_tmux "$target_script" "$LOGFILE"
    ;;
  nohup)
    run_nohup "$target_script" "$LOGFILE"
    ;;
  fg)
    run_foreground "$target_script" "$LOGFILE"
    ;;
  *)
    echo "Mode de lancement inconnu: $LAUNCH_MODE"; exit 1 ;;
esac

echo "==> Done. Monitor logs in $LOG_DIR"
echo "Si tu veux que je convertisse un top5 en JSON, relance : $0 --convert-top5 path/to/top5_v8_23_gen_080.npz"
