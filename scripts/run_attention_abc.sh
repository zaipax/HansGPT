#!/usr/bin/env bash
# Start independent tmux jobs; never overwrite an attempted run or occupy a busy GPU.
set -euo pipefail
cd /root/HansGPT
export PATH="/root/.local/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
stage=${1:?smoke, full or worker}
round=${2:?round identifier}
[[ "$round" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
data=/root/HansGPT/data/processed/modelscope_zhwiki_full_v1
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty source checkout'; exit 2; }
[[ "$(git branch --show-current)" == main ]] || exit 2

if [[ "$stage" == worker ]]; then
    variant=${3:?variant}
    mode=${4:?mode}
    case "$variant" in a) gpu=4;; b) gpu=5;; c) gpu=6;; *) exit 2;; esac
    [[ "$mode" == full || "$mode" == smoke ]] || exit 2
    export CUDA_VISIBLE_DEVICES="$gpu"
    if uv run --frozen python -m hansgpt_research.train_attention_glyph_lm \
        --config "configs/experiments/hansgpt_attention_${variant}.json" \
        --data "$data" --run-name "$round" --mode "$mode"; then
        printf '0\n' > "artifacts/logs/${round}.exit_code"
    else
        code=$?
        printf '%s\n' "$code" > "artifacts/logs/${round}.exit_code"
        exit "$code"
    fi
    exit 0
fi

[[ "$stage" == smoke || "$stage" == full ]] || exit 2
command -v tmux >/dev/null
mkdir -p artifacts/logs
# Check all three assignments before starting any job.
for variant in a b c; do
    case "$variant" in a) gpu=4;; b) gpu=5;; c) gpu=6;; esac
    run="${round}_${variant}"
    [[ "$stage" == smoke ]] && run="${run}_smoke"
    [[ ! -e "artifacts/logs/${run}.console.log" && ! -e "artifacts/logs/$run" ]] || {
        echo "Run already attempted: $run"; exit 2;
    }
    if tmux has-session -t "$run" 2>/dev/null; then echo "Session already exists: $run"; exit 2; fi
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ "$used" -lt 100 ]] || { echo "GPU $gpu is occupied"; exit 2; }
done
if [[ "$stage" == full ]]; then
    uv run --frozen python scripts/verify_attention_smokes.py --round "$round"
fi
for variant in a b c; do
    run="${round}_${variant}"
    [[ "$stage" == smoke ]] && run="${run}_smoke"
    tmux new-session -d -s "$run" \
        "bash scripts/run_attention_abc.sh worker $run $variant $stage >artifacts/logs/${run}.console.log 2>&1"
    echo "Started $run"
done
