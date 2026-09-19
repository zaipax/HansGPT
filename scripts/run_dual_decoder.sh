#!/usr/bin/env bash
set -euo pipefail
cd /root/HansGPT
export PATH="/root/.local/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5
mode=${1:?smoke or full}
round=${2:?round name}
[[ "$round" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ -z "$(git status --porcelain)" && "$(git branch --show-current)" == main ]] || exit 2
if [[ "$mode" == worker ]]; then
  mode=${3:?training mode}
  [[ "$mode" == smoke || "$mode" == full ]] || exit 2
  if uv run --frozen python -m hansgpt_research.train_attention_glyph_lm \
      --config configs/experiments/hansgpt_dual_decoder.json \
      --data data/processed/modelscope_zhwiki_full_v1 --run-name "$round" \
      --mode "$mode" --smoke-tokens 32768; then
    printf '0\n' > "artifacts/logs/$round.exit_code"
  else
    code=$?
    printf '%s\n' "$code" > "artifacts/logs/$round.exit_code"
    exit "$code"
  fi
  if [[ "$mode" == full ]]; then
    uv run --frozen python scripts/evaluate_attention_abc.py --variant D --run-name "$round" \
      --output "artifacts/reports/${round}_evaluation" \
      >"artifacts/logs/${round}_evaluation.log" 2>&1
  fi
  exit 0
fi
[[ "$mode" == smoke || "$mode" == full ]] || exit 2
command -v tmux >/dev/null
run="$round"
[[ "$mode" == smoke ]] && run="${round}_smoke"
[[ ! -e "artifacts/logs/$run" && ! -e "artifacts/logs/$run.console.log" ]] || exit 2
used=$(nvidia-smi -i 5 --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "$used" -lt 100 ]] || { echo 'GPU5 is occupied'; exit 2; }
if [[ "$mode" == full ]]; then
  uv run --frozen python - "$round" <<'PY'
import json, subprocess, sys
from pathlib import Path
from hansgpt_research.train_glyph_lm import sha256
from hansgpt_research.train_structured_glyph_lm import effective_config
p=Path('artifacts/logs')/(sys.argv[1]+'_smoke')
r=json.loads((p/'training_complete.json').read_text())
m=json.loads((p/'metadata.json').read_text())
c=json.loads(Path('configs/experiments/hansgpt_dual_decoder.json').read_text())
assert r['status']=='complete' and r['mode']=='smoke' and r['progress']['optimizer_steps']>0
assert r['metadata_sha256']==sha256(p/'metadata.json')
assert m['git_commit']==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert m['config']==effective_config(c,mode='smoke',smoke_tokens=32768)
assert m['model_family']=='dual_decoder_v1' and m['cuda_visible_devices']=='5'
assert m['data_manifest_sha256']==c['data_requirements']['manifest_sha256']
print('Dual-decoder current-source smoke verified')
PY
fi
mkdir -p artifacts/logs
tmux new-session -d -s "$run" \
  "bash scripts/run_dual_decoder.sh worker $run $mode >artifacts/logs/$run.console.log 2>&1"
echo "Started $run on physical GPU5"
