# HansGPT C-Qwen1.5B

This experiment scales the current pure Transformer C task model to the dense
Qwen3 1.7B shape while keeping the task objective and data unchanged.

## Outer language transformer

| setting | value |
| --- | ---: |
| hidden size | 2048 |
| decoder layers | 30 |
| query heads / KV heads | 16 / 8 |
| head dimension | 128 |
| SwiGLU intermediate size | 6144 |
| RMSNorm epsilon | 1e-6 |
| RoPE theta | 1,000,000 |
| query/key head RMSNorm | enabled, before RoPE |
| attention and MLP bias | disabled |
| attention pattern | full causal |

The Q/K normalization uses the Hugging Face Qwen3 attention implementation,
but the surrounding model still accepts only `inputs_embeds`. The vocabulary
embedding is removed, and no text token IDs or character IDs enter the model.

## Task-specific modules

The 4x4 patch `AttentionGlyphEncoder` remains width 128, four layers, and four
heads. Each outer position is still one 32x32 binary glyph. The byte decoder
remains four causal layers with inner width 256, eight heads, and FFN width 768.
It emits 128 bytes with 256 classes per byte, which are unpacked MSB-first into
the 32x32 target. Generated binary grids are fed back to the same glyph
encoder; there is no glyph-bank lookup, OCR, or candidate projection.

The model has 1,515,243,008 trainable parameters with the default C decoder.
The training target remains byte categorical cross-entropy, reported as
nats/pixel after division by 1024, over `chinese_document_v3` packed windows.

## GPU benchmark

The four-rank benchmark uses one real context-1024 window per rank, head chunk
128, FP16, gradient checkpointing, xFormers attention, and fused AdamW. The
timed region includes the outer forward/backward, byte-head loss, NCCL gradient
all-reduce, and optimizer update. Input preparation is done before timing.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  scripts/benchmark_qwen3_c_multigpu.py \
  --output artifacts/logs/qwen3_c_1p5b_gpu0_3.json
```

The output records the source commit, verified dataset hashes, model parameter
count, per-rank peak memory, and global valid-targets/second. GPUs 4-7 are not
selected by this command.
