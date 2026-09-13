# Diagnosis of comma-only byte generation

## Main evidence

The current eight-GPU model is the same C architecture, but equal 100M-position
budgets did not provide equal optimizer update counts:

| | Old C final | Current C final |
| --- | ---: | ---: |
| Successful positions | 100,003,725 | 100,000,000 |
| Updates | 30,494 | 1,546 |
| Mean valid positions/update | 3,279.5 | 64,683.1 |
| Training context | 256 | 1024 |
| Global sequence batch | 32 | 64 |
| Data | Short Wikipedia documents | Packed multidomain Chinese v3 |
| LR horizon | 100M positions | Full 1,089,139,385-position pass |

Most old document batches had much less than 256 valid positions per sequence;
new packing fills windows. World size and context also increased. Thus each
current update consumes about 19.7 times as many targets. GPU parallelism itself
does not reduce updates when global batch is fixed; the global batch changed.

Crucially, the retained old `generation_step00001505.npz` also contains one
repeated bitmap, **pixel-identical to the current comma**. Its validation NLL
was 0.0103249 at 1505 updates, versus current 0.0103424 at 1546 updates. The
validation sets differ, so NLL equality is supporting historical evidence,
not a controlled generalization comparison. Old final NLL was 0.0035446.

## Controlled tests on the current checkpoint

`scripts/diagnose_byte_comma.py` replays saved outputs and runs fixed probes.
Eight original prompts produce the same next glyph with native SDPA and
xFormers; cached and uncached native decoding agree on the checked prompts.

For 32 held-out packed-window targets, vary only how much left context is
provided, ending at the same target:

| Context length | Greedy next output | Paired target NLL/pixel |
| --- | --- | ---: |
| 16 | 32/32 commas | 0.0081423 |
| 64 | 32/32 commas | 0.0081800 |
| 256 | 32/32 commas | 0.0081830 |
| 1024 | 32/32 commas | 0.0081688 |

Longer inference context does not remove the symptom. This does **not** isolate
the causal effect of training at context1024; that requires matched retraining.

At context1024, shuffling real prefix states changes mean target NLL from
0.0081688 to 0.0083760 (+2.54%). Only 19/32 targets worsen. Distinguishing
different real contexts is weak in this probe. Zeroing the state gives 0.436623,
but zero is out of distribution and cannot by itself prove semantic usage.

For eight contexts, the comma also has lowest joint glyph NLL among ten common
diagnostic candidates. This is not solely a local-greedy implementation artifact;
the learned conditional distribution currently favors the comma. Candidate
scoring is diagnostic only and never replaces model feedback with a font lookup.

## Sampling reveals partial glyph learning

Sampling one next glyph for each of 32 contexts yields 25 exact content glyphs
at temperature0.7, 14 at temperature1, and one at temperature1.2. These tiny
samples are exploratory, not an optimized decoding protocol or OCR accuracy.

Free-running sampling at temperature1 on eight fixed prompts produces 256
raw grids: 122 exact content glyphs (47.66%), one other control, and adjacent
repeat rate 0.40%. Original greedy outputs were all commas. Visual inspection
shows many recognizable glyphs alongside malformed glyphs and incoherent text.
Sampling exposes other modes; it does not establish fluent language generation.

## Next experiment

Prioritize optimization/sample efficiency, rather than assuming context1024 is
broken or raising temperature as a cure. Keep context1024 and the full-corpus
LR schedule; compare global batch8 (one sequence per GPU) against global batch64
on the same successful-position prefix. A 10M-position comparison would provide
approximately 1220 versus 154 updates. Track wall time, NLL, raw glyph validity,
blank/comma rates, context-shuffle sensitivity, and both greedy/sampled text.
For a direct context ablation, subsequently hold global valid targets/update
approximately fixed while changing context and adjusting sequence batch.

No training or default decoding settings were changed during this diagnosis.
The greedy symptom remains reproducible. Raw reports and samples are under
`artifacts/reports/byte_comma_diagnosis/`; this evidence does not guarantee that
additional training alone restores the old final quality.
