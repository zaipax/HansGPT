# HansGPT 二值字形 GPT 实验协议

本实验直接将每个 32×32 二值格子编码成一个视觉 embedding，以因果语言模型预测下一格的 1024 个二值像素。输入没有字符 ID embedding，输出没有字符分类头。训练使用逐像素 BCE logits，生成时二值化并反馈同一个 CNN。

## 固定实验条件

- 数据：ModelScope `AI-ModelScope/wikipedia` 的 `20231101.zh` 六个分片，镜像修订和文件 SHA-256 固定在 `configs/datasets/modelscope_wikipedia_zh.json`，内容哈希已与上游固定版本核对。
- 正文：严格汉字和允许标点，固定 OpenCC；完整预处理规则、近重复算法及其召回限制见 `CHINESE_CORPUS_PILOT.md`。按来源页面拆分，测试集不参与阈值和检查点选择。
- 字形：固定 Noto Sans CJK SC Regular 2.004，字号 26、基线 27、画布 32×32、阈值 128，像素只取 0/1；字体内容哈希随数据 manifest 保存。
- 模型：共享 CNN、12 层 768 维 GQA Transformer、1024 像素线性头，默认共 78,118,368 参数，全部随机初始化和联合训练。
- 设备：用户指定 GPU 0。启动使用 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`，先核对占用，再检查 FP16 前向与反向。
- 环境：PyTorch 2.8.0、torchvision 0.23.0，来自项目的官方 CUDA 12.8 wheel 索引。该 PyTorch 构建包含 V100 所需的 `sm_70`；较新的当前构建已移除它。更改仅在项目 uv 环境，不修改驱动、系统 CUDA 或其他项目环境。
- 优化：FP16 autocast 与 GradScaler；AdamW 学习率 3e-4，norm 和 bias 不做权重衰减；按有效目标格子数归一化和累计预算。
- 正式预算：1,500,000,000 个有效下一格目标，初始关闭早停；完整参数见 `configs/experiments/hansgpt_binary_v1.json`。训练集可以重复呈现，报告独立内容量和实际遍历次数，不能把重复量称为新增语料。
- 上下文上限：1024 格。批次内只裁去共同的尾部 padding，保留全部有效目标和文档边界；实际 batch 与累积设置由服务器烟测确定并固化在运行 metadata 中。

## 运行顺序

所有命令在服务器 `/root/HansGPT` 的干净 main 上运行。先按 AGENTS.md 完成 fetch、pull 和 `uv sync --frozen`；若直连 GitHub 不可用，可在一次 SSH 会话内使用既有代理完成 Git 传输，不持久修改 Git 代理设置。数据文件直接由服务器从 ModelScope 下载。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python scripts/check_environment.py --dtype fp16
uv run ruff check .
uv run pytest -q

uv run python -m hansgpt_research.prepare_corpus \
  --shards 2 --max-pages 10000 --max-han 500000 \
  --output data/processed/modelscope_zhwiki_smoke_v2
uv run python scripts/verify_corpus.py data/processed/modelscope_zhwiki_smoke_v2

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python -m hansgpt_research.train_glyph_lm \
  --data data/processed/modelscope_zhwiki_smoke_v2 \
  --run-name hansgpt_binary_v1_smoke --mode smoke --smoke-tokens 32768

uv run python -m hansgpt_research.prepare_corpus \
  --shards all --max-pages 0 --max-han 0 \
  --output data/processed/modelscope_zhwiki_full_v1
uv run python scripts/verify_corpus.py data/processed/modelscope_zhwiki_full_v1

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python -m hansgpt_research.train_glyph_lm \
  --data data/processed/modelscope_zhwiki_full_v1 \
  --run-name hansgpt_binary_v1 --mode full

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python -m hansgpt_research.evaluate_glyph_lm \
  --checkpoint artifacts/checkpoints/hansgpt_binary_v1/best.pt \
  --data data/processed/modelscope_zhwiki_full_v1 \
  --output artifacts/reports/hansgpt_binary_v1
```

以上为操作模板。实际生效的 batch、累积、预算、配置哈希和启动命令随运行记录保存；不能用烟测完成标记代替正式训练完成。长任务使用独立 tmux 会话和日志；断点恢复必须匹配数据、模型和配置，并恢复优化器、GradScaler、随机状态及采样位置。

训练与评估必须检查绑定当前 manifest 和八个模型输入文件 SHA-256 的成功验证凭据；完整训练还检查六个源分片的实际扫描行数和完成状态。验证器独立重渲染全部字形，并对固定抽取的验证／测试段落执行全训练集五字片段重叠审计。抽检覆盖有限，不据此宣称完全没有语义污染。

可选 `--sampler sortish` 将相近长度的文档块组合成批次，以减少 padding；它保留每个样本和目标、确定性的逐轮随机化及恢复游标。启用前先用独立 benchmark 与随机采样比较，并把实际选项固定在正式运行配置中。

## 完成和评估标准

正式训练只有在完成所声明预算、写出终止原因并保存可恢复检查点后才算完成。评估使用验证集选定的最佳检查点及二值阈值，遍历完整测试拆分并核对有效目标总数。

报告至少包括：数据来源与校验和、清洗与拆分统计、模型及参数量、Git 与软件版本、实际训练预算和遍历次数、损失曲线、吞吐和峰值显存、检查点身份，以及未加权 BCE/NLL、前景 F1、IoU、Dice、Hamming、完整点阵匹配和最近字形 Top-1/Top-5。内容字与控制格子分开报告，并列出按训练频次分层的结果。

全背景和训练像素频率预测作为基线。自由续写输出原始 0/1 格子并反馈 CNN，评测 32、128、512 格；最近字形只用于分析与文字导出，不替换生成图。图像碰撞、检索近邻、合法字形和语义正确性是不同概念，不能混用。

这是单字体、按来源文档泛化的语言建模实验。同一个汉字可出现在不同拆分的文档中；其结果不能证明未见字符或未见部件组合泛化。无论效果好坏，完整结果应保留失败现象和数据局限，不以“看起来像汉字”作为成功结论。
