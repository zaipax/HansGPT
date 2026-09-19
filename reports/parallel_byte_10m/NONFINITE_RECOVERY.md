# Nonfinite-loss investigation and guarded continuation

The original 100M pilot failed at 91,571,221 positions. Its in-memory weights
were lost; the latest saved state was at 50M. Therefore the exact historical
cause cannot be established from the retained checkpoint alone.

## Evidence

`scripts/replay_byte_nonfinite.py` reconstructed epoch 0, sampler cursor 90040
and the same batch of four packed context-1024 windows. Using the retained 50M
weights, all checked encoder/backbone activations and parameters were finite.
The NLL was 0.16731406 for eager FP16, 0.16731445 for compiled FP16 and
0.16731298 for eager FP32. This did not reproduce the historical fault and
does not prove that the later weights were healthy. The report is
`artifacts/reports/parallel_byte_50m_failure_replay.json` on the server.

The ranked hypotheses were FP16 intermediate overflow, a compiled-path
numerical problem, and corrupted weights. A deterministic regression fixture
then exercised the real ByteBackward path with finite FP32 decoder parameters
whose projection exceeded FP16's range. Both eager and compiled old paths
produced NaN loss. This establishes the overflow failure class, not the exact
cause of the lost 91.57M state.

## Recovery behavior

The single-GPU configuration enables `nonfinite_fp32_retry`. Encoder output,
backbone output and each byte-loss chunk are checked before unsafe gradients
can reach an optimizer step. On a nonfinite forward value, discard partial
gradients, restore Torch RNG, and recompute the same data and mask using eager
FP32 with a byte-head chunk of at most 128. Only a successful optimizer step
advances the data cursor or target counter. Existing AMP gradient-overflow
retries remain in place; the normal path is still compiled FP16.

If FP32 also fails, save the CPU batch, selected mask and progress under
`nonfinite_batch.pt`, plus a separately marked diagnostic model/optimizer/RNG
snapshot. Diagnostic snapshots cannot be passed to normal resume. This keeps
failure evidence separate from known-good checkpoints. Distributed precision
retry is deliberately rejected until a coordinated multi-rank protocol exists.

The recovery config is
`configs/experiments/hansgpt_qwen3_parallel_byte_gpu3_100m_recovery.json`.
Architecture, optimizer, batch, data and LR schedule are unchanged. Continue
from the original 50M checkpoint to 100M, checkpoint every 10M, keep one recent
checkpoint and the final milestone. The original 50M source is preserved in
its separate run directory. Normal output directories use experiment suffix
`hansgpt_qwen3_parallel_byte_gpu3_100m_recovery_v2`.

## Validation and disk space

The two overflow regressions failed on the pre-fix code. After the fix, 29
focused tests passed, covering compiled/eager recovery, FP32 gradient agreement,
clean batches, refusal of NaN weights, resume, masking and retention. A further
test passed for refusing diagnostic snapshots as normal resume inputs.

A separate `--smoke --smoke-fp32` run on GPU3 exercises the actual 1.5B model,
restored AdamW/scaler state, batch 4 and context 1024 in full FP32. Smoke weights
are not used for the formal continuation. Its console log is
`artifacts/logs/parallel_byte_recovery_smoke.console.log`.

The full-FP32 smoke at `df89164` completed 65,536 additional positions in 17
updates with no new AMP overflow (the counter retained the source checkpoint's
two earlier retries). Peak allocation was 26.4599 GiB. Its temporary checkpoint
was removed after successful validation/generation. Formal guarded continuation
was launched from source commit `b63dd2e` in tmux `parallel-byte-gpu3-recovery`,
console `artifacts/logs/parallel_byte_gpu3_recovery.console.log`, and status
`artifacts/logs/hansgpt_qwen3_parallel_byte_gpu3_100m_recovery_v2_full/status.json`.
The new run resumes the original 50M checkpoint, not the FP32 smoke weights.
At 100M it automatically runs the same nine inference conditions on GPU3 into
`artifacts/reports/parallel_byte_gpu3_100m_recovery_inference_v2/`.

The filesystem had only 3.7 GiB free. The completed 10M smoke checkpoint was
removed. With explicit user approval, the serial GPU4-7 run's 300M, 400M, 500M,
600M and 700M checkpoints were removed, reclaiming about 85 GiB. Its 100M,
200M, 800M, 900M and 1B checkpoints, all logs/reports, and active training
were retained. Available space recovered to approximately 106 GiB.
