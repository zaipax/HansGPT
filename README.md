# 汉字点阵排字板

## 当前字形预训练入口

当前主线为纯 Transformer C 版：patch 字形编码器、24 层 GPT 主干和
128 步自回归字节解码器。正式预训练使用 `scripts/train_byte_multigpu.py`；
当前八卡配置为 `configs/experiments/hansgpt_byte_c_eight_gpu_global_lr_100m.json`。
**学习率按完整训练集的全局有效位置数调度，试跑停止预算不改变调度周期**。
当前运行完整一遍训练数据前 1 亿位置，详见 [07_byte_glyph_pretraining.md](docs/07_byte_glyph_pretraining.md)。
**默认加速方案固定为 xFormers＋torch.compile＋PyTorch fused AdamW**；
新入口缺少任一组件会报错，不会自动降级。compile 覆盖完整字节头损失，
主干保持 eager；当前每卡 bsz=8、ctx=1024、head chunk=2048，不启用重计算。
`scripts/train_byte_glyph.py` 及旧恒定学习率配置仅用于历史吞吐对照，
不代表正式预训练调度。

一个零依赖的实时汉字点阵前端项目。左侧输入文字，右侧立即转换成 32×32 点阵字形。
完整技术演进、实验报告与系统设计索引详见 [docs/README.md](docs/README.md)：

- **字形探测研究**：冻结语言模型（Qwen3.5-2B）隐藏层线性解码 32×32 点阵与瓶颈诊断 → [`01_glyph_probe_research.md`](docs/01_glyph_probe_research.md)
- **语料与数据管线**：中文维基/多领域文本严格清洗、32×32 矢量点阵渲染与长文档因果打包 → [`02_corpus_and_data_pipeline.md`](docs/02_corpus_and_data_pipeline.md)
- **原生二值字形生成**：端到端预测下一格 32×32 二值像素模型（GlyphGPT v1/v2）与生成诊断 → [`03_binary_glyph_lm.md`](docs/03_binary_glyph_lm.md)
- **架构演进与双解码器**：A/B/C 注意力对比、语义与空间双解码器及字形编解码器修复 → [`04_architecture_exploration.md`](docs/04_architecture_exploration.md)
- **条件字形 VAE 体系**：基于连续潜在空间的整图生成与大规模参数搜索 → [`05_conditional_glyph_vae.md`](docs/05_conditional_glyph_vae.md)
- **底层硬件加速调优**：Tesla V100 优化技巧、Apex、CUDA 图与 xFormers 算子极限标定 → [`06_hardware_and_kernel_tuning.md`](docs/06_hardware_and_kernel_tuning.md)
- **字节预训练与规模化**：当前纯 Transformer C 主线、多卡分布式调度、全局余弦衰减与 1.5B 扩展 → [`07_byte_glyph_pretraining.md`](docs/07_byte_glyph_pretraining.md)

## 功能

- 左右分栏的输入与预览界面
- 每个汉字使用真实 32×32 Canvas 绘制
- 右侧每行显示 20 个汉字位，行数不设上限
- 字位根据输入内容动态创建和删除，不预生成空白画布
- 连续输入自动每 20 字换行，也支持使用回车提前换行
- 只重绘发生变化的字位，适合实时输入
- 窄屏设备自动切换为上下布局，点阵画布支持横向滚动

## 运行

直接用浏览器打开 `index.html` 即可，无需安装依赖或执行构建命令。
