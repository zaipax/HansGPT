# Eight-GPU byte Transformer with full-corpus LR scheduling

This supersedes the constant-LR four-GPU run at the user's request. Start from
the original random seed; do not resume weights trained under the superseded
schedule. GPUs 0–7 use per-rank batch8, global batch64, context1024, head2048,
accumulation1, no backbone recomputation. Pure C architecture and default
xFormers/compiled byte head/PyTorch fused AdamW remain unchanged.

## Separate schedule horizon from pilot budget

The checksum-pinned `chinese_document_v3` training split contains exactly
**1,089,139,385 valid prediction positions**. The schedule horizon is one full
training pass, excluding validation/test, padding and masked transitions.
The pilot stops at **100,000,000 global successful positions** (9.18% of that
pass); it uses exactly the LR values a full-pass run would use at those positions.

Chosen full-pass protocol: linear warmup over 1% of training positions,
rounded to 10,891,394; peak LR 0.0003; then cosine decay to 0.00003 at the
end of the full training pass. The peak is not multiplied by GPU count.

| Global successful position | LR |
| --- | ---: |
| 1,000,000 | 0.00002754468 |
| 10,000,000 | 0.00027544683 |
| 10,891,394 | 0.00030000000 |
| 50,000,000 | 0.00029912453 |
| 100,000,000 (pilot stops) | 0.00029547556 |
| 1,089,139,385 (full pass ends) | 0.00003000000 |

`schedule_total_positions` and `warmup_positions` control the schedule;
`target_tokens` controls stopping only. The runner verifies the full horizon
against the actual training dataset. Tests assert that pilot/full-run LR
prefixes agree and that changing world size does not change LR at a fixed
global position count. Legacy constant-LR configs retain their old semantics.

## Execution and artifacts

Config: `configs/experiments/hansgpt_byte_c_eight_gpu_global_lr_100m.json`.
Use `uv run torchrun --standalone --nproc_per_node=8 scripts/train_byte_multigpu.py
--config configs/experiments/hansgpt_byte_c_eight_gpu_global_lr_100m.json`, with
the established xFormers PYTHONPATH, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `NCCL_P2P_DISABLE=1`, `NCCL_CUMEM_HOST_ENABLE=0`.

Retain ten checkpoints at 10M increments. Sharding, exact position budgets,
weighted post-backward gradient synchronization and checkpoint replica checks
remain as in the four-GPU runner. Eight-rank smoke precedes the formal launch.
The checkpoints retain per-rank RNG and global progress for future recovery;
the current runner still has no resume CLI. A future resume must preserve the
global schedule horizon and restore progress rather than restart warmup.

Artifacts use `hansgpt_byte_c_eight_gpu_b8_h2048_global_lr_100m_v1_full` under
`artifacts/logs`, `artifacts/reports` and `artifacts/checkpoints`. Server tmux
session: `byte-eight-global-100m`, socket `hansgpt-lr-v1`.
