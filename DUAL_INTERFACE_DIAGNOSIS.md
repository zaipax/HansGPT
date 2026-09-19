# Glyph reconstruction interface diagnosis

## Scope

GPU5 replayed the B arm's reconstruction stage from the unchanged r1 best
checkpoint, then extended reconstruction from 2000 to 10000 updates at the same
3e-4 learning rate and batch 256. No language optimization was performed.
The 10707 training and 1189 held-out reconstruction bitmaps use the same seeded,
alias-grouped split as the pilot. Held-out here means reconstruction-held-out,
not unseen by the original language model.

The one-step smoke reproduced the original validation NLL. The full replay
reproduced B's 2000-step reconstruction losses and post-warm-up validation NLL.
Intermediate module checkpoints now preserve the temporary adapter, unlike the
original pilot, allowing future inspection without another replay.

## Counterfactual module swaps

The original GPT and semantic decoder stay fixed. Only the glyph encoder and
spatial decoder are exchanged. All cases use the same 128 validation chunks.

| Components entering the original language path | After 2000 reconstruction updates | After 10000 |
|---|---:|---:|
| Original encoder, original spatial decoder | 0.31773 | 0.31773 |
| Changed encoder, original spatial decoder | 0.39164 | 0.38638 |
| Original encoder, changed spatial decoder | 0.49706 | 2.47526 |
| Changed encoder, changed spatial decoder | 0.61239 | 2.61787 |

Both interfaces cause degradation; the decoder-side mismatch dominates after
longer reconstruction. The decoder learns conditions from a temporary mapping
of the glyph embedding, while language inference provides a different semantic
state. The encoder also changes while its downstream GPT remains fixed. Simply
discarding the adapter and reconnecting the old language path is not a valid
transfer procedure. Retaining an adapter alone would not prove alignment either:
the glyph and semantic condition distributions must explicitly be reconciled.

## Reconstruction capacity and generalization

| Updates | Training exact match | Training mean pixel errors | Held-out exact match | Held-out mean pixel errors | Held-out NLL |
|---|---:|---:|---:|---:|---:|
| 2000 | 7 / 10707 (0.065%) | See raw report | 0 / 1189 | See raw report | 0.17247 |
| 6000 | 691 / 10707 (6.45%) | 5.72 | 0 / 1189 | 65.21 | 0.37883 |
| 10000 | 2235 / 10707 (20.87%) | 2.35 | 0 / 1189 | 62.76 | 0.48882 |

At 10000 updates, foreground F1 is 0.99482 on training glyphs and 0.86088 on
held-out glyphs. Exact match is strict, so this is not evidence that every held-out
image is visually unrecognizable. Nevertheless, the generalization gap is large;
held-out NLL deteriorates as reconstruction becomes more confident. More updates
alone do not establish a successful compositional glyph codec.

The encoder currently compresses 64 patch tokens into one width-128 summary,
projects it to 1024, then derives four conditioning slots. This is an architectural
observation, not proof that 128 dimensions cannot represent the glyphs. Whether
retaining spatial information improves held-out reconstruction requires a control.

## Recommended next experiments

1. Keep the original model intact. Diagnose the glyph codec separately before
   another large language run, using identical glyph splits and meaningful
   held-out F1, Hamming error, and exact-match metrics.
2. Frozen-encoder control: retain the original encoder and train a permanent
   adapter plus spatial decoder. Test whether the original representation contains
   enough stroke information without damaging the GPT input interface.
3. Spatial-information control: compare the single summary against a small set
   of learned latent queries attending to patch tokens. Do not use raw-pixel skip
   connections, character IDs, or gallery projection to make reconstruction appear
   successful. Compare compute budgets and parameter counts explicitly.
4. Once a codec works, freeze it and define one shared latent interface. Train the
   semantic side to supply that interface, then consider limited joint tuning.
   Do not swap newly learned glyph modules into an unchanged old GPT and assume
   compatibility. Changing the encoder requires input alignment or retraining too.
5. Treat next-character ambiguity as a separate problem. Latent MSE can also
   average alternatives; codec reconstruction alone does not solve conditional
   language generation or the factorized pixel-output limitation.

Artifacts: `artifacts/reports/dual_interface_diagnosis/report.json` and the
2000/10000-step module snapshots in that directory. The formal r1 and pilot
checkpoints were not overwritten.
