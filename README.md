# 汉字点阵排字板

一个零依赖的实时汉字点阵前端项目。左侧输入文字，右侧立即转换成 32×32 点阵字形。

仓库同时包含冻结语言模型隐藏状态到 32×32 汉字点阵的研究代码。完整方案见
[`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)，首轮 Qwen3.5-2B 实验结果见
[`EXPERIMENT_REPORT_QWEN35_2B.md`](EXPERIMENT_REPORT_QWEN35_2B.md)，完整问题归因见
[`GLYPH_BOTTLENECK_DIAGNOSTIC_REPORT.md`](GLYPH_BOTTLENECK_DIAGNOSTIC_REPORT.md)。

以 32×32 二值字形为输入、经共享视觉编码器与因果模型预测下一格 1024 个二值像素的方案见
[`HANSGPT_MODEL_AND_DATA_DESIGN.md`](HANSGPT_MODEL_AND_DATA_DESIGN.md)。该文档是下一阶段设计提案。

服务器上的中文维基下载、严格中文过滤和 32×32 二值字形数据准备流程见
[`CHINESE_CORPUS_PILOT.md`](CHINESE_CORPUS_PILOT.md)。

GPU 0 上从头训练、断点恢复与完整评估的固定实验协议见
[`BINARY_GPT_EXPERIMENT.md`](BINARY_GPT_EXPERIMENT.md)。

首轮二值模型的生成问题分析、相关论文与下一轮改进实验方案见
[`BINARY_GLYPH_GENERATION_RESEARCH.md`](BINARY_GLYPH_GENERATION_RESEARCH.md)。

以完整中文句子为目标的混合像素头、条件对抗微调和原图反馈诊断协议见
[`BINARY_GPT_V2_EXPERIMENT.md`](BINARY_GPT_V2_EXPERIMENT.md)。

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
