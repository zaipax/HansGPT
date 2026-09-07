# 中文维基二值字形语料试制

这是固定来源的 CPU 数据管线，输出纯汉字和允许标点的段落，以及每个 Token 对应的 32×32 二值格子。它用于验证 HansGPT 数据和训练接口，不代表已完成正式训练集的全部质量审查。

所有命令在服务器 `/root/HansGPT` 的干净 main 分支运行。按 AGENTS.md 拉取代码并执行 `uv sync --frozen` 后：

```bash
uv run python scripts/check_environment.py --cpu
uv run pytest -q tests/test_prepare_corpus.py
uv run python -m hansgpt_research.prepare_corpus \
  --snapshot 20260901 \
  --max-han 5000000 \
  --output data/processed/zhwiki_binary_pilot_v1
uv run python scripts/verify_corpus.py data/processed/zhwiki_binary_pilot_v1
```

长任务使用 tmux，日志写入 `artifacts/logs/`。下载器支持断点续传，核对维基官方大小和 SHA-1，再记录 SHA-256；字体直接来自 Noto Sans CJK SC Regular 2.004 官方发布。已存在的输出目录不会被覆盖，复跑应指定新版本目录。源码修改在本地 worktree 完成并推送，服务器只拉取和运行。

处理规则：

- 只解析正文命名空间，排除重定向，保存页面与修订 ID、来源链接。
- 去除模板、引用、公式、表格和媒体引用时保留断开边界；不把断开的正文拼成一句话。
- NFC、固定 OpenCC `t2s` 和显式标点映射；只保留汉字与允许标点。
- 夹杂数字、外文、空格或其他符号的段落整段拒绝，不进行未经审核的数字或术语改写。
- 默认保留至少 40 个汉字、至多 4096 个字符的段落，去除明显重复字符、精确重复段落及不可渲染段落。
- 同一来源页面固定在一个拆分中，按哈希约 98%/1%/1% 分配；实际比例和数量写入 manifest。
- 固定字号 26、基线 27、32×32 画布与阈值 128；检测空白和裁切，记录像素碰撞。最终像素只取 0/1。

输出目录包含：

| 文件 | 用途 |
|---|---|
| `train/validation/test.parquet` | 中文正文、来源修订、哈希和拆分元数据 |
| `train/validation/test.txt` | 可阅读的正文；空行只是导出分隔，不用于隐式跨段训练 |
| `glyph_bank.npz` | `uint8[N,32,32]`，元素仅为 0 或 1 |
| `glyph_inventory.json` | 字形资产索引、控制格子和碰撞分组 |
| `*.uint16`、`*.offsets.npy` | 每段的 BOS、正文格子、EOS 序列；字节序为 little-endian |
| `manifest.json` | Git 提交、来源与字体校验和、规则、版本、统计和输出校验和 |
| `verification.json` | 独立读取产物后的文本、二值像素、索引、拆分和精确去重检查 |

索引只用于读取图像资产。模型输入时还原像素，下一格像素作为监督；不能把索引直接作为字符 embedding 或字符分类目标。未用的 PAD 和 NEWLINE 控制图也保留在字形库中，使用频次与内容字符应分开统计。

当前限制明确记录在 manifest：只使用官方首个文章分片并按 dump 顺序扫描；严格字符检查不能代替统计语言识别；未完成近重复聚类、外部评测去污染和全面人工质量抽检。正式扩量前应完成这些步骤，并保留逐页面归属与适用许可。任何字形碰撞不能作为可区分字符身份处理。
