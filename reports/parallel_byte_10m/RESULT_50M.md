# Parallel byte model: interrupted 100M continuation and 50M evaluation

The GPU3 continuation stopped at 91,571,221 valid positions, step 22,512,
with `Nonfinite global loss` at 2026-09-16 23:08:09 UTC (September 17 07:08
China time). Five AMP overflow retries had been recorded. The failing batch
was not committed as an update. The precise cause of the nonfinite loss has
not been diagnosed; the error alone does not establish weight corruption.

The latest saved checkpoint is **50M**, not 91.57M or 100M. The automatic
100M evaluation did not run because training exited unsuccessfully.

- Training source commit: `6361609`.
- Checkpoint: `artifacts/checkpoints/hansgpt_qwen3_parallel_byte_gpu3_100m_v1_full/positions_050000000.pt`.
- SHA256: `4f31fad6214da9e85e6b490d8ed9ba86cd738e159cdb7b76f2e72842c5e45418`.
- Evaluation source commit: `1084bca` (documentation-only change after training source).
- Reports: `artifacts/reports/parallel_byte_gpu3_50m_inference_v1/`.
- Local ignored copy: `artifacts/parallel_byte_gpu3_50m_inference_v1/`.

On September 17, the saved checkpoint was evaluated on physical GPU3 with the
same eight validation documents, seed, prompt lengths 8/128/512, and 256 raw
generated glyphs per document as the 10M evaluation. All nine conditions
completed. Feedback uses raw grids, without inventory projection. GPU4-7's
separate training run continued throughout.

| Prompt length | Greedy exact content membership | Sampling T=0.7 | Sampling T=1.0 | Greedy next-256 exact reference match |
| --- | ---: | ---: | ---: | ---: |
| 8 | 25.59% | 0% | 0% | 0.88% |
| 128 | 23.93% | 0.049% | 0% | 0.93% |
| 512 | 36.33% | 0.098% | 0% | 1.22% |

All conditions had zero EOS. Greedy next-256 foreground F1 ranged from 0.2405
to 0.3387. The fixed teacher-context validation NLL improved from 0.22115571 at
10M to 0.15715600 nats/pixel at 50M, on the identical 5,517 targets. Free-running
inventory membership measures bitmap validity, not correct continuation.

Visual inspection of greedy and T=0.7 contact sheets shows some recognizable
initial characters, but continued fragmentation, black blocks, punctuation
loops and repeated partial glyphs. All eight greedy short-prompt samples
triggered the short-cycle flag. Sampling produces mostly noisy composite
glyphs. Thus more training improved loss and greedy glyph validity, but did
not produce normal readable generation at 50M. No 100M conclusion is available.

Measured aggregate rollout throughput was 179-189 glyphs/s at batch eight;
this is not a matched serial-decoder speed benchmark. Before resuming toward
100M, diagnose the nonfinite loss and recover from the saved 50M state. The
91.57M in-memory state was lost when the worker exited.
