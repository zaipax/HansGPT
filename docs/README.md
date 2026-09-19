# HansGPT 研究与技术文档总览

本目录汇总了 HansGPT 项目从冻结大模型字形探测，到原生二值字形模型、双解码器/CVAE 架构探索，再到当前**纯 Transformer C 字节预训练主线与 15 亿参数扩展**的完整研究历程与技术报告。

为便于查阅与长期维护，原有 40 余篇分散的实验记录与方案已系统性归纳合并为 **7 篇专题技术文档**：

---

## 文档导航

### [1. 汉字字形探针研究 (`01_glyph_probe_research.md`)](01_glyph_probe_research.md)
探针研究理论基础、实验协议与表征瓶颈分析：
- **总体规划**：汉字点阵与大语言模型中间状态线性可分性理论假设（`RESEARCH_PLAN.md`）
- **实验报告**：Qwen3.5-2B 冻结大模型中间隐藏状态字形解码实验（`EXPERIMENT_REPORT_QWEN35_2B.md`）
- **瓶颈诊断**：线性探针泛化瓶颈、词表覆盖与隐藏层表征退化归因报告（`GLYPH_BOTTLENECK_DIAGNOSTIC_REPORT.md`）

---

### [2. 中文语料与数据处理管线 (`02_corpus_and_data_pipeline.md`)](02_corpus_and_data_pipeline.md)
从原始非结构化文本到点阵训练张量的数据工程体系：
- **维基语料下载与清洗**：严格繁简转换、标点与汉字白名单过滤，32×32 Noto Sans 确定性渲染（`CHINESE_CORPUS_PILOT.md`）
- **多领域语料构建**：文学叙事、文言历史与网络高质量文本（`CHINESE_MULTIDOMAIN_CORPUS.md`）
- **连续性重构**：打破短段落碎片化，多进程上下文保持与重复过滤（`CONTIGUOUS_CORPUS.md`）
- **长文档因果打包**：GPT 风格的 EOS 因果流窗口打包与目标有效损失掩码规范（`DOCUMENT_PACKING.md`）

---

### [3. 原生二值字形生成模型 v1 与 v2 (`03_binary_glyph_lm.md`)](03_binary_glyph_lm.md)
完全摆脱 Unicode / BPE，以 32×32 二值点阵作为原生自回归预测目标的初代探索：
- **系统架构提案**：CNN 视觉编码器 + Causal Transformer + 像素级 BCE 头（`HANSGPT_MODEL_AND_DATA_DESIGN.md`）
- **v1 全量实验**：15 亿目标训练、学习曲线与完整留出集评估报告（`BINARY_GPT_EXPERIMENT.md`）
- **生成退化诊断**：背景像素占空比失衡、局部笔画断裂与生成质量诊断（`BINARY_GLYPH_GENERATION_RESEARCH.md`）
- **v2 进阶方案**：整图伯努利混合分布、条件对抗微调 (GAN) 协议（`BINARY_GPT_V2_EXPERIMENT.md`）

---

### [4. 模型架构探索：A/B/C 对比与双解码器 (`04_architecture_exploration.md`)](04_architecture_exploration.md)
视觉编码与解码架构的横向系统对比：
- **A/B/C 架构对比**：Patch 注意力 vs CNN 编码器、像素头 vs 自回归字节解码器对比（`ATTENTION_ABC_EXPERIMENT.md`, `ATTENTION_ABC_R2.md`）
- **双解码器方案**：语义 Transformer 与空间 Transformer 解码器联合预测（`DUAL_DECODER_EXPERIMENT.md`, `DUAL_DECODER_DIAGNOSIS.md`）
- **消融与接口修复**：采样有效性、因果接口漂移诊断与不可变编解码器定义（`DUAL_ABLATION_PILOT.md`, `DUAL_INTERFACE_DIAGNOSIS.md`, `GLYPH_CODEC_REPAIR.md`）

---

### [5. 条件字形变分自编码器体系 (`05_conditional_glyph_vae.md`)](05_conditional_glyph_vae.md)
基于连续潜在空间的整图一次性生成与重构探索：
- **CVAE 设计与 24 层主干**：共享潜在空间条件变分自编码器架构与字形可读性（`CONDITIONAL_VAE_EXPERIMENT.md`, `CVAE_24L_EXPERIMENT.md`）
- **显存与硬件调优**：GPU 显存压力 Smoke、利用率瓶颈诊断与图编译优化（`CVAE_GPU7_SMOKE.md`, `GPU7_UTILIZATION_DIAGNOSIS.md`, `CVAE_GPU5_OPTIMIZATION.md`）
- **大规模搜索与评测**：8 卡并发学习率搜索、4 卡 1 亿位置训练与后验重构 vs 先验生成差距分析（`CVAE_LR_SEARCH.md`, `CVAE_FOUR_GPU_THROUGHPUT.md`, `CVAE_FOUR_GPU_100M.md`）

---

### [6. 底层加速与硬件调优 (`06_hardware_and_kernel_tuning.md`)](06_hardware_and_kernel_tuning.md)
针对 Tesla V100S PCIe 服务器与算子实现的深度极限压榨：
- **V100 优化全景**：NVIDIA Apex 融合 Adam、CUDA 图与显存池优化对比（`V100_TRICKS_RESULTS.md`）
- **xFormers 算子与 Batch 扩展**：CUTLASS 注意力显存扩展边界与 OOM 临界标定（`XFORMERS_BATCH_SCALING.md`）
- **Head Chunk 分块消融**：分块尺寸对吞吐与反传显存的精准权衡（`XFORMERS_HEAD_SCALING.md`, `XFORMERS_H2048_BATCH_TUNING.md`）

---

### [7. Transformer C 字节预训练与 1.5B 扩展 (`07_byte_glyph_pretraining.md`)](07_byte_glyph_pretraining.md)
**当前主线**：基于纯 Transformer C 结构与 128 步字节解码器的端到端预训练体系：
- **单卡吞吐基准**：Context-1024 默认三件套（xFormers + 编译头 + 融合 AdamW）实现（`BYTE_C_CTX1024.md`, `BYTE_C_B4_MEMORY.md`）
- **关键问题排查**：空白输出成因排查与“逗号死锁循环”的优化步数饥饿归因（`BYTE_BLANK_DIAGNOSIS.md`, `BYTE_COMMA_DIAGNOSIS.md`）
- **多卡分布式调度**：4 卡 / 8 卡训练基准、权重同步与断点管理（`BYTE_C_FOUR_GPU.md`, `BYTE_C_FULL_CORPUS.md`）
- **全局余弦学习率调度**：基于 10.89 亿全语料有效预测位置的全局调度协议（`BYTE_C_EIGHT_GPU_GLOBAL_LR.md`）
- **密集模型规模化**：15 亿参数 Qwen3 结构（30 层、Q/K-Norm、xFormers）扩展方案（`QWEN3_DENSE_C_1P5B.md`）
