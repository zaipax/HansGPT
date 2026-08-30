# 汉字字形可解码性研究方案

## 1. 项目摘要

本项目研究一个冻结预训练语言模型的隐藏状态中，是否包含可以被简单输出层恢复的汉字字形信息，以及原生多模态预训练是否会增强这种信息。

模型主体、Token Embedding 和所有中间层均保持冻结，只移除或绕过原始词表输出头，新增一个预测 32×32 二值点阵的输出头：

```text
中文输入
  ↓
冻结的预训练模型
  ↓
目标位置隐藏状态 h ∈ Rᵈ
  ↓
Linear(d, 1024)
  ↓
reshape(32, 32)
  ↓
Sigmoid + 二值化阈值
```

第一阶段研究“输入汉字能否恢复未见汉字的字形”；第二阶段研究“纯中文语义上下文能否预测下一个汉字的字形”。

## 2. 核心研究问题

1. 纯文本预训练模型的隐藏状态中是否存在可线性解码的汉字字形信息？
2. 这种信息是字符身份记忆，还是能够泛化到未参与输出头训练的汉字？
3. 原生多模态预训练是否使文本隐藏状态包含更多汉字字形信息？
4. 更大的汉字 Token 覆盖是否会提高字形恢复能力？
5. 字形信息主要存在于浅层、中间层还是最后一层？
6. 在只提供中文语义上下文、目标汉字不出现在输入中时，模型能否生成合理字形？

## 3. 研究假设

- H1：冻结模型的隐藏状态可以恢复训练过的汉字点阵，但未见汉字表现会明显下降。
- H2：对于部件已见但组合未见的汉字，模型表现优于完全未见部件的汉字。
- H3：Qwen3.5 的原生多模态训练和更大词表会提升字形可解码性。
- H4：在两个 tokenizer 都将目标汉字编码为单 Token 的交集上，Qwen3.5 仍会优于 Qwen3；若差异消失，提升主要来自 tokenizer 覆盖。
- H5：普通像素准确率会高估模型能力，前景 F1、IoU 和最近汉字检索准确率更能反映真实表现。

## 4. 模型选择

### 4.1 开发模型

[Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)

- 用于跑通 tokenizer 分析、数据管线、隐藏向量提取和输出头训练。
- 文本隐藏维度为 2048，共 24 层。
- 与正式主模型使用相同的 Qwen3.5 混合架构，能提前发现兼容性问题。

### 4.2 正式主模型

[Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)

- 语言模型参数约 4B，完整模型包含视觉编码器，共约 5B 参数。
- 文本隐藏维度为 2560，共 32 层。
- 词表大小为 248,320。
- 架构为 Gated DeltaNet 与全注意力混合结构。
- 官方模型为原生多模态 Causal Language Model with Vision Encoder。
- Apache-2.0 许可证。

本项目只输入文本。视觉编码器不参与前向计算，可放在 CPU，或只加载 `text_config` 对应的语言部分。多模态预训练可能已经影响文本隐藏状态，这正是需要测量的研究变量。

### 4.3 关键对照模型

[Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base)

- 纯文本因果语言模型。
- 文本隐藏维度同样为 2560，共 36 层。
- 词表大小为 151,936。
- Apache-2.0 许可证。

Qwen3 与 Qwen3.5 的隐藏维度相同，因此可使用参数量完全一致的 `Linear(2560, 1024)` 输出头。两者对比可以测量多模态训练、架构更新和 tokenizer 扩展带来的综合变化。

### 4.4 暂不使用 Qwen3.8

[Qwen 官方当前的 Qwen3.8 型号](https://huggingface.co/Qwen/models?search=Qwen3.8)主要包括 27B、180B 和更大的 MoE 模型，目前没有适合 16GB 显存的小型 2B/4B Base 型号。

Qwen3.8-27B 的 BF16 权重远超本机显存；INT4 权重本身也接近 16GB 上限，运行缓存和中间张量没有足够空间。量化还会改变隐藏状态，干扰跨模型研究结论。因此第一阶段不使用 Qwen3.8。未来若官方发布 Qwen3.8-4B-Base，可在不改变数据和输出头设计的情况下加入第三组实验。

## 5. 硬件与运行策略

本机 GPU：RTX 5060 Ti 16GB。

推荐设置：

```text
模型主体精度：BF16
输出头精度：FP32
上下文长度：MVP 使用 64，第二阶段测试 128
特征提取 batch：从 2 或 4 开始，逐步测试 8/16
use_cache：false
Attention：PyTorch SDPA
模型主体梯度：关闭
输出头梯度：开启
```

Windows 下优先使用支持 RTX 50 系列和相应 CUDA 架构的 PyTorch 官方构建。第一版不依赖 FlashAttention 和 bitsandbytes，以减少环境和量化变量。所有 Python 环境由 `uv` 管理。

Qwen3.5-4B-Base 完整 BF16 权重约需 10GB 原始存储空间；纯文本语言部分约为 4B 参数。模型主体不反向传播时，16GB 显存足以处理短上下文。输出头和 AdamW 优化器只占用很少显存。

## 6. 输出层与训练目标

对于 Qwen3.5-4B-Base 和 Qwen3-4B-Base：

```text
输入维度：2560
输出维度：1024
权重数量：2560 × 1024
偏置数量：1024
总参数量：2,622,464
```

输出的每一维代表 32×32 点阵中的一个位置。训练时输出 Logit，不在模型内部提前二值化：

```python
logits = glyph_head(hidden_state)
logits = logits.reshape(-1, 32, 32)
probabilities = logits.sigmoid()
binary_bitmap = probabilities >= threshold
```

主损失建议采用：

```text
加权 BCEWithLogitsLoss + Dice Loss
```

前景笔画像素少于背景像素，必须根据训练集统计设置正样本权重。二值化阈值由验证集选择，不能直接假设 0.5 最优。

## 7. 数据下载清单

### 7.1 现在必须下载

#### A. Unicode Unihan 数据

- 下载地址：[Unihan.zip](https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip)
- 官方说明：[Unicode Han Database（UAX #38）](https://www.unicode.org/reports/tr38/)
- 使用条款：[Unicode Copyright、Terms of Use and Licenses](https://www.unicode.org/copyright.html)
- 建议保存位置：`data/raw/unicode/Unihan.zip`

用途：建立 Unicode 汉字清单，并取得读音、部首、总笔画、简繁异体关系等元数据。Unicode Data Files 当前适用 Unicode License v3。

#### B. Noto Sans CJK SC 字体

- 精确下载：[08_NotoSansCJKsc.zip（Sans 2.004，约 90MB）](https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/08_NotoSansCJKsc.zip)
- 版本页面：[Noto Sans CJK Version 2.004](https://github.com/notofonts/noto-cjk/releases/tag/Sans2.004)
- 项目仓库：[notofonts/noto-cjk](https://github.com/notofonts/noto-cjk)
- 字体许可证：[SIL Open Font License 1.1](https://github.com/notofonts/noto-cjk/blob/main/Sans/LICENSE)
- 建议保存位置：`data/raw/fonts/`

解压后保留简体中文 Sans 的 Regular 字重 OTF 文件。不要使用仓库的通用 `releases/latest` 链接，因为当前最新发布页指向 Noto Serif CJK。实际实验必须记录字体文件名、版本号和 SHA-256，确保所有点阵可复现。

#### C. 中文维基百科文章语料

- 下载目录：[zhwiki latest dump](https://dumps.wikimedia.org/zhwiki/latest/)
- MVP 推荐：下载目录中第一个编号的 `zhwiki-latest-pages-articles-multistream1...xml...bz2` 文章分片，当前大小约 250MB。
- 完整实验：`zhwiki-latest-pages-articles-multistream.xml.bz2`，当前压缩后约 3.5GB。
- 校验文件：同目录中的 `zhwiki-latest-sha1sums.txt` 或 `zhwiki-latest-md5sums.txt`。
- 建议保存位置：`data/raw/zhwiki/`
- 许可说明：[Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)

MVP 只需要第一个文章分片，不建议一开始下载完整 3.5GB 文件。维基文本通常按照 CC BY-SA 4.0/GFDL 提供；保存和发布处理后数据时必须保留来源、页面标识和归属信息。

### 7.2 后续可选下载

#### D. 中文维基词典

- 下载目录：[zhwiktionary latest dump](https://dumps.wikimedia.org/zhwiktionary/latest/)
- 推荐文件：`zhwiktionary-latest-pages-articles-multistream.xml.bz2`
- 建议保存位置：`data/raw/zhwiktionary/`

用途：构造“纯中文释义/读音描述 → 汉字点阵”任务。它不是第一阶段必需数据。

#### E. CHISE IDS 汉字构件数据

- 仓库：[chise/ids](https://github.com/chise/ids)
- 建议保存位置：`data/raw/chise-ids/`

用途：获得 `⿰`、`⿱` 等表意文字描述序列，构建“部件已见、组合未见”的严格拆分。使用或再分发前需要单独核对仓库内对应数据文件的许可证。

### 7.3 暂时不需要下载

- 英文释义为主的 CC-CEDICT：不满足第一阶段纯中文语料目标。
- 未明确允许再分发的网络词典抓取数据。
- 完整 Common Crawl/WuDao 级别大语料：第一阶段规模过大且许可、清洗成本高。
- 多套字体：第一阶段固定一个字体，后续再加入字体外泛化实验。

## 8. 数据集 A：HanziGlyph-8K

### 8.1 目标

建立约 7,000～10,000 个汉字与固定 32×32 二值字形之间的映射，用于冻结模型的字形可解码性实验。

```text
输入：清
目标：清在 Noto Sans CJK SC Regular 中的 32×32 二值点阵
```

### 8.2 字符选择

1. 从 Unihan 获得候选字符。
2. 使用中文维基语料统计字频。
3. 只保留 Noto Sans CJK SC 有真实字形的字符。
4. 排除控制字符、兼容字符异常、缺字方框和不可稳定渲染字符。
5. 保留常用字、低频字和生僻字三个频率层次。
6. 分别记录简体、繁体、异体和 Unicode 扩展区。

### 8.3 确定性点阵生成

- 使用固定字体文件，而不是系统字体名称。
- 使用固定版本的 FreeType/Pillow 渲染。
- 固定字号、画布、基线、水平垂直居中规则。
- 先渲染灰度图，再使用固定阈值生成二值图。
- 保存渲染器版本、字体 SHA-256、阈值和边界规则。
- 检测所有字符是否误渲染成相同的 tofu 缺字框。

### 8.4 数据表

`characters.parquet`：

```text
char_id
char
codepoint
frequency
radical
stroke_count
pronunciation
variant_group
qwen3_token_ids
qwen35_token_ids
qwen3_single_token
qwen35_single_token
font_supported
bitmap_packed
split_random
split_component
split_frequency
```

1024 个二值点压缩后为 128 字节，不需要为每个样本保存 PNG。

## 9. 数据集 B：ChineseContext-Glyph

### 9.1 目标

构建纯中文上下文到下一个汉字点阵的预测任务：

```text
输入：河水十分清
目标字符：澈
监督信号：澈的 32×32 二值点阵
```

目标字符不得出现在当前位置的输入中。用于严格未见字符实验时，还应排除输入前缀中出现同一个目标字符的样本。

### 9.2 MVP 规模

```text
样本数量：200,000
目标汉字：3,000～5,000
每字上下文：尽量保持 20～100 条
上下文长度：32～128 Token，首轮固定 64
```

确认存在有效信号后，再扩展到 1,000,000 条样本。

### 9.3 清洗规则

- 解析维基 XML，只保留正文。
- 删除模板、HTML、URL、引用、表格、代码和导航内容。
- 使用 NFC 规范化；避免会改变兼容汉字的过度规范化。
- 使用 OpenCC 转换为简体，同时保存原始文本和转换版本。
- 中文汉字和中文标点占比建议至少 85%。
- 删除乱码、超长重复、广告和近重复段落。
- 按来源文章划分数据，避免同一文章跨训练集和测试集。
- 对高频字设置每字采样上限，避免“的、是、了”等字符支配数据集。
- 不对低频字简单复制过采样；优先寻找更多独立上下文或在损失中加权。

### 9.4 数据表

`contexts.parquet`：

```text
sample_id
input_text
target_char_id
source_project
source_page_id
source_revision_id
source_title
context_length
split_document
split_target_char
```

上下文数据只引用 `target_char_id`，不要重复存储点阵。

## 10. Tokenizer 与隐藏状态对齐

对全部候选汉字生成 tokenizer 报告：

- 单 Token 字符。
- 多 Token 字符及 Token 数量。
- 简繁体和生僻字覆盖。
- Qwen3 与 Qwen3.5 单 Token 字符交集。

主报告至少包含两组结果：

1. Native：使用每个模型自己的全部有效字符。
2. Intersection：只使用在两个 tokenizer 中均为单 Token 的字符。

多 Token 汉字可采用“最后一个子 Token 隐藏状态”和“所有子 Token 平均池化”两种方式，并作为独立消融实验，不能与单 Token 结果混在一起。

## 11. 数据拆分与防泄漏

### 11.1 随机字符拆分

按字符而不是样本拆分：

```text
训练字符：70%
验证字符：15%
测试字符：15%
```

同一个目标汉字的所有上下文只能出现在一个集合中。

### 11.2 构件组合拆分

利用 IDS 将测试集构造成：

- 语义部件已见、声旁已见、组合未见。
- 部分部件未见。
- 整体结构类型已见与未见。

该拆分用于区分“字符查表”与“构件组合泛化”。

### 11.3 字频拆分

- 高频训练、低频测试。
- 高频/中频/低频分桶报告。
- 字频由本次清洗后的中文维基语料重新统计，不依赖不可复现的第三方频率表。

### 11.4 上下文拆分

另外建立一个目标字符可重叠、但来源文档不重叠的普通上下文拆分，用于测量标准的上下文预测能力。该结果不能代替字符不重叠实验。

## 12. 冻结模型与隐藏向量缓存

模型主体设置为 `eval()`，所有参数 `requires_grad=False`。使用 `torch.no_grad()` 提取目标位置隐藏状态，再与输出头分开训练。

建议先离线缓存隐藏状态：

```text
Qwen3.5-4B hidden_size：2560
BF16 每条隐藏向量：2560 × 2 = 5120 字节
20 万条：约 1.0GB
100 万条：约 5.1GB
```

缓存后可卸载语言模型，只用输出头重复测试不同损失、阈值和数据拆分。隐藏特征应按模型、层号、tokenizer 版本、输入模板和数据版本分目录保存，避免误用。

## 13. 分层探针

除最后一层外，为以下归一化深度训练独立线性头：

```text
Embedding 输出
25% 深度
50% 深度
75% 深度
最后一层
```

每个探针都使用相同数据拆分、相同参数初始化策略和相同训练预算。Qwen3 与 Qwen3.5 层数不同，因此按相对深度比较，不直接按层号比较。

## 14. 对照实验

至少包含：

1. 正确的冻结模型隐藏状态 → 正确点阵。
2. 冻结随机初始化模型 → 正确点阵。
3. 正确隐藏状态 → 随机打乱的汉字点阵。
4. Token ID Embedding 查表 → 点阵。
5. 原始 LM Head 预测 Token → 查表转换为点阵。
6. 单线性层与两层 MLP 输出头对比。
7. Qwen3 与 Qwen3.5 在单 Token 交集字符集上的对比。

线性输出头是主要结果。MLP 只能作为“是否存在非线性可恢复信息”的补充实验。

## 15. 评测指标

不能只报告普通像素准确率，因为背景点远多于笔画点。

主要指标：

- 前景像素 Precision、Recall、F1。
- Intersection over Union（IoU）。
- Dice 系数。
- 完整 1024 位点阵匹配率。
- Hamming 距离。
- 最近字形检索 Top-1/Top-5 准确率。
- 最近字形是否具有正确部首、IDS 结构和笔画范围。

所有指标按以下分桶分别报告：

- 高频、中频、低频。
- 单 Token、多 Token。
- 简体、繁体、异体。
- 左右、上下、包围、独体等结构。
- 已见部件组合、未见组合、未见部件。

使用 bootstrap 计算置信区间。模型选择和二值化阈值只依据验证集，测试集只运行最终配置。

## 16. 自回归限制

1024 位点阵不能直接作为下一步的 Token 输入。第一阶段的任务是单次探针和下一字符点阵预测，不进行自由运行的长文本生成。

若未来需要连续生成，可以：

1. 将输出点阵与候选字形库做最近邻匹配，得到 Unicode 字符后重新 tokenizer；或
2. 保留原始 LM Head，并同时增加 Glyph Head；或
3. 训练点阵输入编码器，但这会改变输入侧，不再属于“只改输出层”的当前研究。

## 17. 项目工程结构

```text
HansGPT/
├── RESEARCH_PLAN.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── models/
│   ├── datasets/
│   └── experiments/
├── data/
│   ├── raw/                 # 不提交 Git
│   ├── interim/             # 不提交 Git
│   └── processed/           # 仅提交小型清单与元数据
├── src/hansgpt_research/
│   ├── data/
│   ├── glyphs/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── scripts/
├── tests/
├── artifacts/              # 不提交大型权重和隐藏特征
└── web/                     # 当前点阵可视化页面后续迁入
```

所有 Python 命令使用项目 `uv` 环境运行。原始 dump、模型权重、隐藏特征和训练产物必须加入 `.gitignore`。

## 18. 实施阶段

### Phase 0：环境与可行性验证

- 创建 `uv` Python 工程。
- 安装适配 RTX 50 系列的 PyTorch。
- 加载 Qwen3.5-2B-Base。
- 验证纯文本前向、隐藏状态提取和显存占用。
- 生成 100 个汉字点阵并训练最小输出头。

### Phase 1：HanziGlyph-8K

- 解析 Unihan。
- 固定 Noto 字体和渲染参数。
- 生成字符表与 32×32 二值点阵。
- 完成 tokenizer 覆盖报告。
- 建立字符不重叠拆分。
- 运行 Qwen3.5-2B、Qwen3.5-4B 和 Qwen3-4B 线性探针。

### Phase 2：纯中文上下文

- 解析中文维基 MVP 分片。
- 清洗、简体化、去重和句段切分。
- 生成 20 万条上下文样本。
- 缓存三个模型的目标隐藏状态。
- 训练并评估语义到点阵输出头。

### Phase 3：构件与层级分析

- 加入 IDS 构件数据。
- 建立构件组合拆分。
- 训练多个冻结层探针。
- 分析部首、结构、字频和 tokenizer 对结果的影响。

### Phase 4：扩展与论文结果

- 将语料扩展到 100 万条。
- 增加第二字体作为域外测试。
- 加入中文释义到字形任务。
- 固化数据版本、配置、随机种子和结果表。
- 完成可视化页面与论文图表。

## 19. 第一阶段完成标准

第一阶段不是以“生成的字看起来不错”为完成标准，而是必须满足：

- 所有测试汉字未参与输出头训练。
- 正确标签实验显著优于标签打乱和随机模型对照。
- 报告单 Token 交集上的 Qwen3/Qwen3.5 结果。
- 前景 F1、IoU 和最近字形检索准确率具有置信区间。
- 字体、Unicode、模型、tokenizer 和语料版本可复现。
- 失败结果也能够区分“隐藏状态没有字形信息”和“输出头只会记忆训练字”。

## 20. 当前立即执行的下载顺序

1. 下载 `Unihan.zip`。
2. 下载 Noto Sans CJK SC Regular 字体。
3. 下载中文维基百科最新目录中的第一个文章 multistream 分片，而不是完整 dump。
4. 下载完成后记录文件名、下载日期、来源 URL 和 SHA-256。
5. 暂时不要下载中文维基词典、CHISE IDS、完整中文维基 dump 或 Qwen3.8 权重。
6. 数据校验完成后，再创建 `uv` 工程和自动处理脚本。
