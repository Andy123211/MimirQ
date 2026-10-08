# 实验室 Hybrid RAG 小样例

该目录提供一个可复现的小语料问答检索实验，使用合成的实验室制度文本，不连接 Milvus、PostgreSQL、Redis 或 Docker 服务，也不需要 API 密钥。样例复用 MimirQ 已有的本地 BGE-M3 embedding provider、中文 BM25 分词器、RRF 融合评分和本地 BGE cross-encoder reranker；报告会保留文档来源和命中片段作为引用。

这是一个针对 Hybrid RAG 检索链路的独立实验环境，不是完整 MimirQ 平台的 API 或 UI 启动环境。首轮运行需下载 BGE-M3 权重（约 2.27 GB）；如启用重排，还需下载 BGE reranker 权重。优先使用 ModelScope 国内源下载 BGE-M3；请预留足够磁盘空间。

## 流程

```text
合成知识片段
  ├─ BGE-M3 dense 检索 ─┐
  └─ BM25 中文检索 ────┴─ RRF ── BGE reranker ── 引用输出
                                              └─ Hit@5 / MRR@5
```

## 检查样例配置

在仓库根目录运行：

```bash
python scripts/run_lab_hybrid_rag.py --check
```

该检查只验证 JSON schema、引用目标和模型配置，不会导入模型或下载权重。
运行实际评测时，若调用者没有设置，脚本会为当前进程临时生成 JWT 配置校验所需的随机密钥，并将数据库地址设为内存 SQLite；二者都不会写入文件或覆盖已有环境变量。该样例不会启动服务、创建 JWT 或访问数据库。

## 创建独立 Conda 环境

从仓库根目录运行以下命令。环境定义位于本示例目录，名称为 `mimirq-hybrid-lab`，不会修改其他 Conda 环境，也不安装完整后端、数据库或 OCR 依赖：

```bash
conda env create -f examples/lab_hybrid_rag/environment.yml
conda run -n mimirq-hybrid-lab python scripts/run_lab_hybrid_rag.py --check
```

如果已经创建过该环境，可用以下命令同步依赖：

```bash
conda env update -n mimirq-hybrid-lab -f examples/lab_hybrid_rag/environment.yml --prune
```

## 从 ModelScope 国内源下载 BGE-M3

在仓库根目录运行。权重保存在仓库之外的独立目录；如网络中断，可重复运行同一命令尝试续传：

```powershell
conda run -n mimirq-hybrid-lab python scripts/download_lab_bge_m3_modelscope.py
```

自定义目录可添加 `--local-dir D:\models\mimirq-bge-m3`。下载完成后，运行评测时指定模型目录：

```powershell
$env:MIMIRQ_BGE_M3_MODEL_PATH = "D:\tool\model-cache\mimirq-modelscope\bge-m3"
conda run -n mimirq-hybrid-lab python scripts/run_lab_hybrid_rag.py --skip-reranker
```

本地路径模式直接加载 ModelScope 下载的 Sentence Transformers 模型，不会再访问 Hugging Face。网络检查显示 `HF_ENDPOINT=https://hf-mirror.com` 的大权重请求会重定向到 Hugging Face，因此该权重推荐使用 ModelScope。

## 运行本地评测

从仓库根目录使用隔离环境启动本地合成样例。在 Windows PowerShell 中可先将模型缓存定向到空间充足的磁盘：

```powershell
$env:HF_HOME = "D:\models\mimirq-hybrid-lab"
conda run -n mimirq-hybrid-lab python scripts/run_lab_hybrid_rag.py --out runs/lab_hybrid_rag.json
```

若本机已经缓存权重，可在 PowerShell 中离线运行：

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
conda run -n mimirq-hybrid-lab python scripts/run_lab_hybrid_rag.py
```

只评估 BGE-M3 向量检索、BM25 和 RRF 时可加 `--skip-reranker`。命令会打印融合排序与重排后排序的 Hit@5、MRR@5；JSON 报告包含逐题指标、模型名、检索配置和最终引用的 `chunk_id`、来源文件名及片段文本。Hit@5 表示前 5 条中至少命中一个标注相关片段的题目比例；MRR@5 是首个相关片段在前 5 条中的倒数排名的平均值。

该语料与查询仅用于链路冒烟和方法演示，评测数值只能描述这组合成样例，不代表真实实验室知识库或简历项目的质量结论。若要形成项目结果，应替换为经授权的领域语料和人工标注问题，并保存配置、语料版本与运行报告。

## 文件

- `config.json`：模型、RRF、重排、引用和指标参数，以及查询与 gold chunk 标注。
- `knowledge_base.json`：8 条合成实验室制度片段，包含 source metadata。
- `../../scripts/run_lab_hybrid_rag.py`：本地内存检索与评测入口。

## 来源与许可

样例脚本复用 MimirQ 仓库组件并随仓库的 Apache-2.0 许可分发；MimirQ 的第三方组件声明仍以根目录 `NOTICE` 为准。`knowledge_base.json` 是为本示例新写的合成内容，不包含真实单位制度或生产数据。
