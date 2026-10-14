# 医药知识问答系统（解析、分块、索引与问答）

本仓库实现可追溯的药品说明书 PDF 解析、MySQL 落库、父子分块、摘要导航、Qdrant Dense Vector 索引，以及 BM25 + 向量召回、RRF 融合、Cross-Encoder 精排、条件式上下文压缩和基于原文证据的问答。

## 安装

```powershell
cd D:\medical-knowledge\medical-knowledge-qa
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item config.example.yaml config.yaml
```

编辑 `config.yaml`，填写 `mysql_url`、`reranker_model`，并将 `rag_internal_token` 设置为与管理后台相同的令牌。`config.yaml` 已被 Git 忽略。请从项目根目录运行以下命令；原有环境变量只作为缺少 YAML 配置项时的兼容后备。

公共知识库的 PDF 从 `medical-knowledge-admin` 后台导入、审核和发布。QA 工程的 `app.worker` 负责异步解析与索引；新导入文档在发布前保持 `PENDING`，不会进入用户问答。先执行 `python -m app.cli init-db` 应用迁移，再单独运行 `python -m app.worker`。`knowledge_import_dir` 可指定原 PDF 持久化目录，默认 `data/imports`。CLI 的 `parse` 和 `index` 命令保留供离线维护使用。

PDF 审核来源映射、修正批次和版本切换的实现说明见 [`../docs/PDF解析与分块审核_实施记录.md`](../docs/PDF解析与分块审核_实施记录.md)。

## 阶段 2：分块与索引

结构化文档导入 MySQL 时会同步生成以下检索单元：

- 按标准章节、子标题/编号条款、医药主题组织局部 Topic Parent；表格另建 Table Parent 和 Row Group Parent；
- 正文事实、列表项、表格行和有说明文字的图片作为 child chunk；只有超长语义单元才按完整句子回退切分并标记重叠；
- 章节达到 2000 tokens、8 个 Child 或 3 个主题时建立 Section Summary；主题达到 800 tokens 且包含至少两个信息维度时建立 Topic Summary。摘要仅保存导航要点和 `source_chunk_ids`，不作为医学结论的独立证据；
- 每个 chunk 保留 document、drug、product、原始及规范章节、层级路径、topic、parent、原文、page、bbox、元素 ID、切分原因及相邻关系；
- MySQL 的 `chunk_retrieval_ft` ngram FULLTEXT 索引提供中文词法检索；`summary_search` 独立检索摘要，`load_summary_sources` 按来源 ID 回到原始 Child；
- Qdrant 保存原始 Child 和摘要的 Dense Vector，使用 `index_channel` 区分证据与摘要路由。

本地启动 Qdrant（REST 端口 6333）：

```powershell
docker run --name medical-qa-qdrant -p 6333:6333 -p 6334:6334 `
  -v medical-qa-qdrant-data:/qdrant/storage qdrant/qdrant
```

准备 Ollama embedding 模型：

```powershell
ollama pull bge-m3:latest
```

应用迁移并重新导入文档，以生成 chunk：

```powershell
.\.venv\Scripts\python.exe -m app.cli init-db
.\.venv\Scripts\python.exe -m app.cli parse-samples
```

增量索引全部 ACTIVE 文档，并删除 Qdrant 中已经失效的文档版本：

```powershell
.\.venv\Scripts\python.exe -m app.cli index
```

全量删除并重建 collection：

```powershell
.\.venv\Scripts\python.exe -m app.cli rebuild-index
```

只重建指定 ACTIVE 文档：

```powershell
.\.venv\Scripts\python.exe -m app.cli index --document-id DOCUMENT_ID
```

## 查询与检索（第 11 章）

先构建 Qdrant 索引，再安装可选的精排依赖，并配置本地可用的 Cross-Encoder 模型：

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[rerank]'
.\.venv\Scripts\python.exe -m app.cli retrieve --query '达可替尼的腹泻如何调整剂量？'
```

`retrieve` 使用 MySQL 中的药品名与别名解析问题，并按 ACTIVE 文档、药品和章节过滤。人群标签可能缺失于短表格行或剂量段落，因此作为查询线索参与排序，不作为硬排除条件。单药、单明确章节的概括型问题优先查询摘要通道；命中可回溯的章节摘要后直接取得其原始 Child，摘要缺失或不完整时再回退正文召回。其他查询由 BM25 从过滤范围内的原始 Child 计算 Top 20，Qdrant Dense Vector 从状态为 READY、模型和 collection 匹配的文档提供另一组 Top 20。两路名次经 RRF 融合后，由 Cross-Encoder 对 `(问题, 原文候选)` 打分。最终返回 5～8 条原文证据，包含页码、来源引用、两路名次、RRF 分数和精排分数。摘要只用于定位来源，不作为最终证据。

口语问题未识别出标准章节，或首次召回为空/候选过少时，检索器会调用 `qwen3.5:latest` 生成结构化查询改写。原问题与规范问题、扩展问题分别召回，再以原问题更高权重进行 RRF 融合。改写结果必须对齐标准 intent 和章节，并通过药品、数值、等级、否定含义校验；校验失败或 LLM 不可用时自动保留原始检索结果。可在 `config.yaml` 中设置 `query_rewrite_enabled: false` 关闭。检索调试输出和网页压缩步骤中会显示 `query_rewrite` 决策。

多药比较按药品和章节维度独立检索；无证据的维度返回“当前说明书未检索到相关表述”。目前 BM25 为查询时对过滤范围内的 Child 全量评分，适合现阶段数据量；大规模语料需要迁移到专门的 BM25 倒排索引。

## 上下文压缩与问答（第 12～13 章）

本机 Ollama 已准备 `qwen3.5:latest` 时，可以运行：

```powershell
.\.venv\Scripts\python.exe -m app.cli answer --query '达可替尼的腹泻如何调整剂量？'
```

`answer` 先执行第 11 章检索。短问题和短证据直接使用精排 Top 3；展开上下文较大时使用向量裁剪；多药或多条件且候选较多时使用 LLM 候选过滤；长原文只在需要时摘取，并要求摘取片段逐字存在于原始 chunk。`--debug` 会额外输出实际触发的压缩步骤。

回答由 `qwen3.5:latest` 生成，返回 `answer`、`evidence`、`confidence`、`warnings`。来源字段从原始 chunk 元数据填充，模型返回的引用必须包含有效 chunk ID 和逐字原文短句；无法核对的引用、数值或强限制性表述会被拦截。多药和多份说明书分别组织答案，缺少证据时明确说明，不从其他药品外推。

## 本地网页与 API

安装网页依赖，在 `config.yaml` 中设置 `mysql_url` 和 `reranker_model` 后启动：

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[web,rerank]'
.\.venv\Scripts\python.exe -m uvicorn app.web.api:app --host 127.0.0.1 --port 8000
```

VS Code 可在“运行和调试”中选择 `启动 药品咨询RAG系统` 并按 F5 启动。调试配置使用项目 `.venv` 解释器和根目录下的 `config.yaml`；首次使用时请从 `config.example.yaml` 复制，并安装 `.[web,rerank]`。

浏览器打开 `http://127.0.0.1:8000/`。页面可提问、按药品筛选、查看答案引用和完整原文。FastAPI 的 `/docs` 提供接口文档；主要接口为 `POST /api/v1/chat/query`、`GET /api/v1/evidence/{chunk_id}` 和 `GET /api/v1/health`。问答接口只支持当前有效的说明书版本；药品筛选支持数据库中已收录药品名及省略剂型的常用写法。在 `config.yaml` 中设置 `enable_retrieval_debug: true` 后，才开放 `POST /api/v1/retrieval/debug`。

## 评测与优化

[黄金样本](evaluation/golden.json)记录 5 份文档的章节/表格/原文检查、21 道单药检索题、1 道跨药比较题及 7 道回答题（含无答案问题）。可查看每个原文 chunk 的候选信息，扩充人工标注：

当前运行结果与适用范围见 [评测报告](evaluation/report.md)。

```powershell
.\.venv\Scripts\python.exe -m app.evaluation.candidates --summary
.\.venv\Scripts\python.exe -m app.evaluation.candidates --section 用法用量 --limit 4
.\.venv\Scripts\python.exe -m app.evaluation.runner --mode all
```

评测报告写入 `data/evaluation/latest.json`，分别列出解析结构与来源保留率、BM25 / Vector / RRF / Reranker 的 Recall@K、MRR、nDCG、药品/章节命中、表格行命中、跨药污染，以及回答的引用可回溯率、药品归属、关键词覆盖和拒答检查。`--mode parse` 或 `--mode retrieval` 可单独运行；`--answer-limit N` 可先抽测 N 道问答。自动指标只检查可确定的事实；Faithfulness、相关性和引用是否真正支撑结论，报告保留人工审核字段，不把字符串匹配当作医学正确性的证明。当前样本低于规格第 17.4 节的每药目标数量，扩充时须人工核对原 PDF 和 chunk ID。

`config.example.yaml` 中提供 `qdrant_url`、`ollama_url`、`embedding_model` 和
`embedding_dimensions` 等默认配置。远程 Qdrant 的密钥填写在本地 `config.yaml` 的
`qdrant_api_key` 中，不要提交该文件。问答模型单次调用默认超时 120 秒，可通过
`ollama_chat_timeout_seconds` 调整。

依赖为 PyMuPDF、pdfplumber 和 PyMySQL。数据库使用 MySQL 8；在 `config.yaml` 中设置 `mysql_url` 后执行初始化和导入。初始化命令会在数据库不存在时创建它，并使用 `utf8mb4`：

```powershell
.\.venv\Scripts\python.exe -m app.cli init-db
.\.venv\Scripts\python.exe -m app.cli parse-samples
```

如果暂时不需要落库，可以将 `mysql_url` 设为 `null`，先运行 `parse-samples`。每份文件的 `parsed.json`、`report.json`、表格原图和图片写入 `data/output/<document_id>/`。样本路径见 [samples.json](samples.json)，如本机路径不同可编辑清单。相同文件 hash 与解析器版本重复导入时，数据库返回 `already present`；解析器版本变更则保留另一条记录，并让同一源文件的新版解析结果接替旧版解析结果。

## 数据表说明

| 表 | 用途 |
|---|---|
| `drug` | 规范药品实体，以规范化通用名称归并。 |
| `drug_product` | 上市药品商品，以批准文号集合优先识别，可归属于同一个规范药品。 |
| `product_approval_number` | 商品与批准文号的一对多关系，支持多规格说明书。 |
| `drug_alias` | 通用名、商品名、英文名及其他别名，并记录商品和文档来源。 |
| `document` | 一次可审计的说明书解析版本，记录文件 hash、解析器版本、业务版本状态和替代关系。 |
| `section` | 文档章节，保存章节原名、规范名、页码范围和标题 bbox。 |
| `element` | 页面文本元素，包括章节标题、正文、列表、页眉页脚和表内文字。 |
| `parsed_table` | 完整结构化表格，保存表题、Markdown、JSON、分页原图和来源。 |
| `table_cell` | 表格单元格，保存行列坐标、原值、规范值及合并单元格继承关系。 |
| `table_row` | 表格行级单元，保存行 JSON、可检索文本和逐页来源。 |
| `image` | 图片资产，包括化学结构图、曲线图和其他内嵌图片。 |
| `parse_report` | 解析质量报告，包括统计、警告、失败页和待复核状态。 |
| `chunk` | 父子检索分块，保存检索文本、元数据、页码、bbox、parent 和全文索引。 |
| `document_index_state` | 记录 chunk 清单和 Qdrant 向量索引状态，支持失败恢复与幂等重建。 |

所有表和字段均在 MySQL 中配置了中文 `COMMENT`，可以通过 `SHOW CREATE TABLE 表名` 或数据库管理工具直接查看。

### 药品、商品和版本关系

```text
drug（规范通用药品）
└── drug_product（具体上市商品）
    ├── product_approval_number（一个或多个批准文号）
    ├── drug_alias（商品名/英文名/通用名）
    └── document（多个说明书版本，最多一个 ACTIVE）
```

`parse_status` 表示解析质量：`READY` 或 `REVIEW`。`version_status` 表示说明书业务有效性：

- `ACTIVE`：该商品当前使用的说明书版本。
- `SUPERSEDED`：已被较新版本替代的历史版本。
- `WITHDRAWN`：人工标记为撤回或停止使用。
- `PENDING`：没有可靠修订日期，需人工确认是否替代当前版本。

数据库通过 `active_product_id` 生成列和唯一索引保证每个 `drug_product` 最多只有一个 `ACTIVE` 文档。导入带有更新修订日期的版本时，当前版本会转为 `SUPERSEDED`，新版本通过 `supersedes_document_id` 指向旧版本。

`drug.generic_name_normalized` 保存规范通用名，`drug_alias` 保存每份说明书实际出现的通用名、商品名和英文名。导入时优先用任一相同批准文号定位已有 `drug_product`；因此新版本即使使用新的商品名或通用名写法，也会复用原有 `drug`，并将新名称记录为别名。没有共同批准文号时，系统先用规范通用名或已知通用名别名绑定 `drug`，再用规范通用名与商品名组合作为后备商品身份；无法可靠判断的全新名称需要人工维护主数据，避免自动误合并。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

需要运行真实 MySQL 集成测试时，将 `config.test.example.yaml` 复制为 `config.test.yaml` 并填写 `test_mysql_url`。五份 PDF 的解析集成测试耗时较长；未配置测试数据库时会跳过。
运行只读的真实问答链路测试时，还需在 `config.yaml` 中配置 `mysql_url`、`reranker_model`，并在 `config.test.yaml` 中设置 `run_live_e2e: true`，再执行 `python -m unittest tests.test_e2e_live -v`。

### 断点调试

VS Code/Cursor 已配置使用项目的 `.venv`。在测试或业务代码左侧设置断点，打开“运行和调试”，选择下列配置后按 `F5`：

- `Python: 调试当前测试文件`
- `Python: 调试全部单元测试`
- `Python: 调试五份 PDF 集成测试`
- `Python: 调试 MySQL 集成测试`
- `Python: 调试五份 PDF 解析命令`

也可以在测试方法上方点击编辑器提供的 `Debug Test`。调试 MySQL 前，将 `config.test.example.yaml` 复制为不会提交的 `config.test.yaml`，并填写测试数据库连接地址。

不使用 IDE 时，可以从命令行进入 `pdb`：

```powershell
.\.venv\Scripts\python.exe -m pdb -m unittest tests.test_tables.TableTests.test_cross_page_without_repeated_header
```

常用命令：`n` 执行下一行、`s` 进入函数、`c` 继续到下一断点、`p 变量名` 查看变量、`q` 退出。

安装 PDF 依赖后可运行五份真实样本的解析集成测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

使用单独的测试数据库验证 MySQL 建表、中文 JSON 和重复导入：

```powershell
Copy-Item config.test.example.yaml config.test.yaml
# 编辑 config.test.yaml 中的 test_mysql_url
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

坐标使用 PDF point，原点在页面左上角，页码从 1 开始。原值始终保留；规范化结果单独存储。
无文字层且有图像内容的页面会进入 `failed_pages`；空白尾页单独列入 `blank_pages`。有警告的文档会标记为 `REVIEW`。跨页表仅在相邻页、列数和位置匹配且续页内容不像新表头时拼接；报告会提示人工核验。内嵌图片按尺寸初步标记为 `chart`、`glyph_fragment` 或 `unclassified`；矢量化学结构图按页面区域裁剪为 `chemical_structure`。图像不会被当作化学结构数据。表格行、图片与正文均保留文档、章节、页码和 bbox。
