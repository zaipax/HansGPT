# 中文维基二值字形语料

此 CPU 管线从用户指定的 ModelScope 下载中文维基数据，输出纯汉字和允许标点的段落，以及每个 Token 对应的 32×32 二值格子。支持小规模烟测和完整中文快照预处理；训练输入必须还原为像素，预测目标也是下一格的 1024 个二值像素。

## 固定来源

使用 ModelScope 的 `AI-ModelScope/wikipedia`，中文子集 `20231101.zh`，六个 Parquet 文件合计约 1.72 GB。下载地址、文件大小和 SHA-256 固定在 [来源配置](configs/datasets/modelscope_wikipedia_zh.json)，文件修订为 `af6f00069c4a886b60b7b08d80882efde42c5432`。这是上游 `wikimedia/wikipedia` 的镜像；用户因服务器外网连接问题明确选择 ModelScope，管线不会自动回退到 Hugging Face 或其他镜像。

数据已提取为 `id,url,title,text` 格式，不能再次当作 wikitext 解析。页面 ID 和页面 URL 来自原行；上游没有提供文章修订 ID，因此 `source_revision_id` 保留为空，并单独保存快照日期、来源文件和文件修订。来源配置记录上游仓库修订与授权说明。ModelScope 的 Apache-2.0 标签与其上游说明中的 Wikipedia CC BY-SA/GFDL 不一致，不能据此把百科正文声明为 Apache 授权；导出保留页面归属信息，公开再分发前仍需检查页面适用权利。

字体直接来自 [Noto 官方仓库 Sans2.004 标签的 Regular OTF](https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf)，遵循 SIL OFL 1.1，记录下载后的 SHA-256。

## 服务器运行

全部下载、Python 执行和预处理都在服务器 `/root/HansGPT` 的干净 main 分支完成。按 AGENTS.md 拉取代码并执行 `uv sync --frozen` 后，先运行：

```bash
uv run python scripts/check_environment.py --cpu
uv run pytest -q tests/test_prepare_corpus.py
uv run python -m hansgpt_research.prepare_corpus \
  --shards 2 --max-pages 10000 --max-han 500000 \
  --output data/processed/modelscope_zhwiki_smoke_v1
uv run python scripts/verify_corpus.py data/processed/modelscope_zhwiki_smoke_v1
```

烟测通过后，完整扫描全部六个中文分片，不设置字符或页面截断：

```bash
uv run python -m hansgpt_research.prepare_corpus \
  --shards all --max-pages 0 --max-han 0 \
  --output data/processed/modelscope_zhwiki_full_v1
uv run python scripts/verify_corpus.py data/processed/modelscope_zhwiki_full_v1
```

默认原始文件目录为 `data/raw/hansgpt_modelscope_wikipedia/20231101/`，烟测下载的第三个分片会在完整运行时复用并重新校验。下载器支持 `.part` 断点续传，完成后核对固定大小和 SHA-256 再发布文件；已存在的成品也必须通过校验。ModelScope 下载失败会保留 partial 并报错，不更换来源。完整下载完成后开始正文扫描。

长任务用 tmux，日志写入 `artifacts/logs/`。已存在的输出目录不会覆盖，失败后重跑应使用新的输出版本；经过校验的 raw 文件仍能复用。源码修改在本地 worktree 完成并推送，服务器只拉取和运行。旧版官方 XML 分片入口仍可通过 `--source wikimedia --snapshot YYYYMMDD --raw ...` 显式使用，不属于默认路径。

## 清洗、去重和拆分

- NFC、固定 OpenCC `t2s` 和显式标点映射，只保留汉字与允许中文标点。
- 夹杂数字、外文、内部空格或其他符号的段落整段拒绝，不删除中间片段后拼接残句，也不进行未经审核的数字或术语改写。段落首尾空白会去除。
- 默认保留至少 40 个汉字、至多 4096 个字符的段落，拒绝连续十次重复字符、字体不支持、空白或裁切的字形。
- 全局 SHA-256 精确去重；去标点后汉字序列相同的变体也仅保留第一次出现。
- 近重复候选使用去标点汉字文本的五字片段：选八个最小 CRC32 哈希，至少命中两个且长度比不低于约 0.9 才比较；最多检查按共享锚点数排序的前一百条候选，确认真实字符串片段集合 Jaccard ≥ 0.9 后删除后出现的段落。CRC32 仅生成候选，不能单独决定重复。候选超限次数写入统计。
- 近重复检查在全局进行，随后来源页面固定在一个拆分中，以页面 ID 和种子哈希约 98%/1%/1% 分配 train/validation/test；实际数量写入 manifest。
- 固定字号 26、基线 27、32×32 画布、阈值 128。检测空白和裁切，记录像素碰撞；输出仅为 0/1。

近重复候选召回是近似的，不能承诺清除全部转载、改写或语义重复。字符检查也不能代替语言识别：来源为中文维基，仍可能保留全汉字的日语引文等少量情况。严格过滤会系统性损失数字密集、科技外文术语多的段落。快照较旧且领域集中，不代表广域现代中文。没有完成外部评测集去污染或全面人工质量审核。

正文按 Parquet batch 流式读取，保留记录和近重复索引暂存 SQLite，不把完整原始语料装进内存。时间成本以总字符数、每段锚点的数据库索引查询和候选相似度比较为主；相似模板极多的部分会慢一些。SQLite 缓存设为约 64 MiB，另需字符缓存、源 batch、输出偏移和页面集合内存。磁盘需同时容纳原始 Parquet、临时记录索引和最终导出；预处理完成后删除 SQLite 暂存文件。

## 产物与验证

| 文件 | 用途 |
|---|---|
| `train/validation/test.parquet` | 中文正文、页面归属、文件修订、哈希和拆分元数据 |
| `train/validation/test.txt` | 可阅读的正文；空行仅作导出分隔 |
| `glyph_bank.npz` | `uint8[N,32,32]`，元素仅为 0 或 1 |
| `glyph_inventory.json` | 字形资产索引、控制格子、像素碰撞分组 |
| `*.uint16`、`*.offsets.npy` | 每段 BOS、正文格子、EOS 序列，little-endian 索引 |
| `manifest.json` | Git 提交、来源及字体校验和、规则、版本、统计和输出校验和 |
| `verification.json` | 独立读取产物后的文本、二值像素、索引、来源、拆分和去重验证 |

验证器按 batch 读 Parquet，以 memmap 读序列，校验输出 SHA-256、纯中文字符、二值像素、逐段索引到文本的一致性、页面不跨集合、精确及标点变体不重复、文件修订与 manifest 一致、导出计数一致。近重复步骤由专门用例验证算法行为，独立产物验证不宣称穷举近重复检测。

索引只用于读取图像资产，不作为可学习的字符 embedding 或字符分类目标。PAD 和 NEWLINE 控制图虽未出现在正文序列中，也保留在字形库，统计时应与内容字符分开。页面不重叠的语言建模拆分允许相同汉字在训练和测试中出现，不能据此声称未见汉字的组合泛化；字形碰撞也不能作为可区分身份处理。
