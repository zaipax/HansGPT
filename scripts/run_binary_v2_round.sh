#!/usr/bin/env bash
# Launch inside a server-side tmux session; all artifacts remain on the server.
set -euo pipefail

round_name="${1:?Usage: run_binary_v2_round.sh ROUND smoke|diagnose|train|evaluate}"
phase="${2:?Specify smoke, diagnose, train, or evaluate}"
[[ "$round_name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid round name' >&2; exit 2; }
case "$phase" in smoke|diagnose|train|evaluate) ;; *) echo 'Invalid phase' >&2; exit 2 ;; esac

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
[[ -z "$(git status --porcelain)" ]] || { echo 'Checkout must be clean' >&2; exit 2; }
[[ "$(git branch --show-current)" == main ]] || { echo 'Run from main' >&2; exit 2; }
source_commit="$(git rev-parse HEAD)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
uv_bin="${HANSGPT_UV_BIN:-uv}"
data_dir=data/processed/modelscope_zhwiki_full_v1
initial=artifacts/checkpoints/hansgpt_binary_v1/best.pt
log_dir="artifacts/logs/${round_name}_pipeline"
mkdir -p "$log_dir"
[[ ! -e "$log_dir/$phase.started" ]] || { echo 'Phase already attempted; inspect its artifacts' >&2; exit 2; }
printf '%s\n' "$source_commit" > "$log_dir/$phase.git-commit"
date -u +%FT%TZ > "$log_dir/$phase.started"
trap 'exit_code=$?; printf "%s\n" "$exit_code" > "$log_dir/$phase.exit"' EXIT

run_module() {
    [[ "$(git rev-parse HEAD)" == "$source_commit" && -z "$(git status --porcelain)" ]] || {
        echo 'Source changed during pipeline; refusing next stage' >&2
        exit 2
    }
    "$uv_bin" run --frozen python -m "$@"
}

diagnose_checkpoint() {
    checkpoint="$1"
    output="$2"
    prompt_count="$3"
    prompt_lengths="$4"
    thresholds="$5"
    max_new="$6"
    run_module hansgpt_research.diagnose_glyph_generation \
        --checkpoint "$checkpoint" --data "$data_dir" --output "$output" \
        --split validation --prompt-count "$prompt_count" --prompt-lengths "$prompt_lengths" \
        --thresholds "$thresholds" --max-new "$max_new" --seed 20260908 \
        --strategy mode_threshold --near-hamming 4
}

case "$phase" in
    smoke)
        for variant in bce gan mixture4; do
            run_module hansgpt_research.train_structured_glyph_lm \
                --config "configs/experiments/hansgpt_binary_v2_${variant}.json" \
                --data "$data_dir" --init-checkpoint "$initial" \
                --run-name "${round_name}_${variant}_smoke" --mode smoke --smoke-tokens 32768
        done
        diagnose_checkpoint "$initial" "artifacts/reports/${round_name}_d0_smoke" 2 16 0.45 8
        diagnose_checkpoint "artifacts/checkpoints/${round_name}_mixture4_smoke/final.pt" \
            "artifacts/reports/${round_name}_mixture_smoke" 2 16 0.45 8
        ;;
    diagnose)
        diagnose_checkpoint "$initial" "artifacts/reports/${round_name}_d0" 64 16,64 0.3,0.45,0.5 128
        ;;
    train)
        for variant in bce gan mixture4; do
            run_module hansgpt_research.train_structured_glyph_lm \
                --config "configs/experiments/hansgpt_binary_v2_${variant}.json" \
                --data "$data_dir" --init-checkpoint "$initial" \
                --run-name "${round_name}_${variant}" --mode full
        done
        ;;
    evaluate)
        for variant in bce gan mixture4; do
            diagnose_checkpoint "artifacts/checkpoints/${round_name}_${variant}/final.pt" \
                "artifacts/reports/${round_name}_${variant}_final" 64 16,64 0.3,0.45,0.5 128
        done
        ;;
esac
date -u +%FT%TZ > "$log_dir/$phase.completed"
