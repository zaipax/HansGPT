# HansGPT 研究与实验文档索引

本文档按技术发展与实验演进，系统索引 `docs/` 目录下的所有研究报告、技术提案、实验记录与诊断文档。

---

## 目录

1. [基础理论与字形探针](#1-基础理论与字形探针)
2. [语料预处理与文档流打包](#2-语料预处理与文档流打包)
3. [端到端二值模型演进 (v1 / v2)](#3-端到端二值模型演进-v1--v2)
4. [架构探索：A/B/C 对比、双解码器与 CVAE](#4-架构探索abc-对比双解码器与-cvae)
5. [底层加速与硬件调优 (V100 / xFormers)](#5-底层加速与硬件调优-v100--xformers)
6. [纯 Transformer C 预训练主线与参数规模化](#6-纯-transformer-c-预训练主线与参数规模化)

---

## 1. 基础理论与字形探针

研究冻结语言模型（Qwen3.5-2B/4B, Qwen3-4B）中间隐藏状态是否保留可线性解码的 32×32 汉字点阵结构。

| 文档 | 描述 |
| --- | --- |
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | 汉字点阵与大语言模型表征研究总体规划与理论假设 |
| [`EXPERIMENT_REPORT_QWEN35_2B.md`](EXPERIMENT_REPORT_QWEN35_2B.md) | 首轮 Qwen3.5-2B 冻结探针训练与留出字符线性解码评测报告 |
| [`GLYPH_BOTTLENECK_DIAGNOSTIC_REPORT.md`](GLYPH_BOTTLENECK_DIAGNOSTIC_REPORT.md) | 探针泛化瓶颈与中间表征退化归因分析报告 |

---

## 2. 语料预处理与文档流打包

构建严格过滤的高质量纯中文矢量点阵数据集及长文档因果打包流水线。

| 文档 | 描述 |
| --- | --- |
| [`CHINESE_CORPUS_PILOT.md`](CHINESE_CORPUS_PILOT.md) | 中文维基下载、繁简转换、严格白名单过滤与 32×32 点阵渲染初探 |
| [`CHINESE_MULTIDOMAIN_CORPUS.md`](CHINESE_MULTIDOMAIN_CORPUS.md) | 多领域（文学、文言、网络精选）纯中文语料清洗与 Parquet 导出 |
| [`CONTIGUOUS_CORPUS.md`](CONTIGUOUS_CORPUS.md) | 连续中文段落上下文保持与多进程去重规范化 |
| [`DOCUMENT_PACKING.md`](DOCUMENT_PACKING.md) | EOS 分隔的长文档流因果打包（GPT-style Packing）与损失掩码规则 |

---

## 3. 端到端二值模型演进 (v1 / v2)

摆脱 Unicode / BPE，以 32×32 二值像素图直接作为输入与预测目标的早期探索。

| 文档 | 描述 |
| --- | --- |
| [`HANSGPT_MODEL_AND_DATA_DESIGN.md`](HANSGPT_MODEL_AND_DATA_DESIGN.md) | 原生二值字形 GPT 总体设计方案与像素预测目标提案 |
| [`BINARY_GPT_EXPERIMENT.md`](BINARY_GPT_EXPERIMENT.md) | 首轮二值模型训练协议、断点恢复与留出集评估 |
| [`BINARY_GLYPH_GENERATION_RESEARCH.md`](BINARY_GLYPH_GENERATION_RESEARCH.md) | 二值字形生成问题诊断、像素稀疏性与背景占空比研究 |
| [`BINARY_GPT_V2_EXPERIMENT.md`](BINARY_GPT_V2_EXPERIMENT.md) | 混合伯努利头、条件 GAN 判别器微调与原图反馈评测协议 |

---

## 4. 架构探索：A/B/C 对比、双解码器与 CVAE

对比 Patch 注意力与 CNN 编码器，探索全图一次性生成、条件变分自编码器等架构。

| 文档 | 描述 |
| --- | --- |
| [`ATTENTION_ABC_EXPERIMENT.md`](ATTENTION_ABC_EXPERIMENT.md) | A（Patch+像素）、B（CNN+字节）、C（Patch+字节）三分支首轮对比 |
| [`ATTENTION_ABC_R2.md`](ATTENTION_ABC_R2.md) | A/B/C 第二轮训练协议与纯 Transformer 方案确立 |
| [`DUAL_DECODER_EXPERIMENT.md`](DUAL_DECODER_EXPERIMENT.md) | 语义与空间双 Transformer 解码器联合一次预测整图方案 |
| [`DUAL_DECODER_DIAGNOSIS.md`](DUAL_DECODER_DIAGNOSIS.md) | 双解码器输出不完整与语义退化归因 |
| [`DUAL_ABLATION_PILOT.md`](DUAL_ABLATION_PILOT.md) | 双解码器参数消融与采样有效性实验 |
| [`DUAL_INTERFACE_DIAGNOSIS.md`](DUAL_INTERFACE_DIAGNOSIS.md) | 编码器与解码器接口漂移诊断 |
| [`GLYPH_CODEC_REPAIR.md`](GLYPH_CODEC_REPAIR.md) | 空间查询编码器修复与不可变编解码器接口规范 |
| [`CONDITIONAL_VAE_EXPERIMENT.md`](CONDITIONAL_VAE_EXPERIMENT.md) | 共享连续字形潜在空间的条件 VAE 方案 |
| [`CVAE_24L_EXPERIMENT.md`](CVAE_24L_EXPERIMENT.md) | 24 层 CVAE 主干训练与多级字形可读性评测 |
| [`CVAE_GPU7_SMOKE.md`](CVAE_GPU7_SMOKE.md) | 完整 CVAE 显存压力与吞吐试跑 |
| [`GPU7_UTILIZATION_DIAGNOSIS.md`](GPU7_UTILIZATION_DIAGNOSIS.md) | 单卡利用率低与梯度反传瓶颈分析 |
| [`CVAE_BATCH_COMPARISON.md`](CVAE_BATCH_COMPARISON.md) | 匹配样本顺序的 batch 规模对比分析 |
| [`CVAE_GPU5_OPTIMIZATION.md`](CVAE_GPU5_OPTIMIZATION.md) | CVAE 头部优化与局部图编译加速验证 |
| [`CVAE_10M_GPU7_EVALUATION.md`](CVAE_10M_GPU7_EVALUATION.md) | 1000 万字符级 CVAE 留出集重构与先验生成差距评估 |
| [`CVAE_LR_SEARCH.md`](CVAE_LR_SEARCH.md) | 8 卡独立并发学习率网格搜索报告 |
| [`CVAE_FOUR_GPU_THROUGHPUT.md`](CVAE_FOUR_GPU_THROUGHPUT.md) | 4 卡同步 CVAE 吞吐与扩展边界测试 |
| [`CVAE_FOUR_GPU_100M.md`](CVAE_FOUR_GPU_100M.md) | 4 卡 1 亿位置 CVAE 训练与后验/先验不一致性结论 |

---

## 5. 底层加速与硬件调优 (V100 / xFormers)

针对服务器硬件（Tesla V100S PCIe）与特定模型结构的计算调优与显存极限探测。

| 文档 | 描述 |
| --- | --- |
| [`V100_TRICKS_RESULTS.md`](V100_TRICKS_RESULTS.md) | Apex 融合优化器、CUDA 图及显存池对比记录 |
| [`XFORMERS_BATCH_SCALING.md`](XFORMERS_BATCH_SCALING.md) | xFormers 显存扩展边界与显存不足 (OOM) 临界点探测 |
| [`XFORMERS_HEAD_SCALING.md`](XFORMERS_HEAD_SCALING.md) | Head chunk 大小对训练吞吐与显存利用率的消融测试 |
| [`XFORMERS_H2048_BATCH_TUNING.md`](XFORMERS_H2048_BATCH_TUNING.md) | 固定 head chunk 2048 时的极限 batch 标定记录 |

---

## 6. 纯 Transformer C 预训练主线与参数规模化

当前项目的核心主线：基于 Patch 注意力编码器、Llama/Qwen3 语言主干与 128 步字节解码器的自回归预训练。

| 文档 | 描述 |
| --- | --- |
| [`BYTE_C_CTX1024.md`](BYTE_C_CTX1024.md) | Context-1024 默认加速（xFormers+编译头+融合AdamW）单卡基准 |
| [`BYTE_BLANK_DIAGNOSIS.md`](BYTE_BLANK_DIAGNOSIS.md) | 字节模型空白输出原因排查与历史差分诊断 |
| [`BYTE_COMMA_DIAGNOSIS.md`](BYTE_COMMA_DIAGNOSIS.md) | 逗号循环死锁问题归因：优化步数饥饿与条件弱化证实 |
| [`BYTE_C_B4_MEMORY.md`](BYTE_C_B4_MEMORY.md) | Batch-4 稳定预训练显存曲线与吞吐确认 |
| [`BYTE_C_FOUR_GPU.md`](BYTE_C_FOUR_GPU.md) | 4 卡 1000 万/1 亿位置字节模型训练与吞吐基准 |
| [`BYTE_C_EIGHT_GPU_GLOBAL_LR.md`](BYTE_C_EIGHT_GPU_GLOBAL_LR.md) | **全局有效位置数调度协议**（10.89 亿位置总周期，解耦试跑停止预算） |
| [`BYTE_C_FULL_CORPUS.md`](BYTE_C_FULL_CORPUS.md) | 4 卡全语料 10.8 亿位置训练记录与断点保留策略 |
| [`QWEN3_DENSE_C_1P5B.md`](QWEN3_DENSE_C_1P5B.md) | **15 亿参数密集 C 模型扩展方案**（30层、Q/K-Norm、xFormers与多卡基准） |
