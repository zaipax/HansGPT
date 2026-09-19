# 汉字点阵排字板

## 当前字形预训练入口

当前主线为纯 Transformer C 版：patch 字形编码器、24 层 GPT 主干和
128 步自回归字节解码器。正式预训练使用 `scripts/train_byte_multigpu.py`；
当前八卡配置为 `configs/experiments/hansgpt_byte_c_eight_gpu_global_lr_100m.json`。
**学习率按完整训练集的全局有效位置数调度，试跑停止预算不改变调度周期**。
当前运行完整一遍训练数据前 1 亿位置，见 [全局调度协议](docs/BYTE_C_EIGHT_GPU_GLOBAL_LR.md)。
**默认加速方案固定为 xFormers＋torch.compile＋PyTorch fused AdamW**；
新入口缺少任一组件会报错，不会自动降级。compile 覆盖完整字节头损失，
主干保持 eager；当前每卡 bsz=8、ctx=1024、head chunk=2048，不启用重计算。
`scripts/train_byte_glyph.py` 及旧恒定学习率配置仅用于历史吞吐对照，
不代表正式预训练调度。单卡测试记录见 [BYTE_C_CTX1024.md](docs/BYTE_C_CTX1024.md)。

一个零依赖的实时汉字点阵前端项目。左侧输入文字，右侧立即转换成 32×32 点阵字形。
完整技术演进与实验报告索引见 [docs/README.md](docs/README.md)。

仓库同时包含冻结语言模型隐藏状态到 32×32 汉字点阵的研究代码。完整方案见
[`RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)，首轮 Qwen3.5-2B 实验结果见
[`EXPERIMENT_REPORT_QWEN35_2B.md`](docs/EXPERIMENT_REPORT_QWEN35_2B.md)，完整问题归因见
[`GLYPH_BOTTLENECK_DIAGNOSTIC_REPORT.md`](docs/GLYPH_BOTTLENECK_DIAGNOSTIC_REPORT.md)。

以 32×32 二值字形为输入、经共享视觉编码器与因果模型预测下一格 1024 个二值像素的方案见
[`HANSGPT_MODEL_AND_DATA_DESIGN.md`](docs/HANSGPT_MODEL_AND_DATA_DESIGN.md)。该文档是下一阶段设计提案。

服务器上的中文维基下载、严格中文过滤和 32×32 二值字形数据准备流程见
[`CHINESE_CORPUS_PILOT.md`](docs/CHINESE_CORPUS_PILOT.md)。

GPU 0 上从头训练、断点恢复与完整评估的固定实验协议见
[`BINARY_GPT_EXPERIMENT.md`](docs/BINARY_GPT_EXPERIMENT.md)。

首轮二值模型的生成问题分析、相关论文与下一轮改进实验方案见
[`BINARY_GLYPH_GENERATION_RESEARCH.md`](docs/BINARY_GLYPH_GENERATION_RESEARCH.md)。

以完整中文句子为目标的混合像素头、条件对抗微调和原图反馈诊断协议见
[`BINARY_GPT_V2_EXPERIMENT.md`](docs/BINARY_GPT_V2_EXPERIMENT.md)。

纯 Transformer 与 CNN 输入的 A／B／C 第二轮训练及完整测试评估见
[`ABC 第二轮效果报告`](reports/attention_abc_r2/REPORT.md)：字节解码方案能生成清晰字形，
但自由续写仍存在严重的词句循环和语义问题。

语义与字形两个 Transformer decoder 联合一次预测完整位图的方案、GPU5 参数测速及
训练协议见 [`DUAL_DECODER_EXPERIMENT.md`](docs/DUAL_DECODER_EXPERIMENT.md)。

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
