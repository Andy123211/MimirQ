# 实验室 Hybrid RAG 小样例

该目录提供一个可复现的小语料问答检索实验，使用合成的实验室制度文本，不连接 Milvus、PostgreSQL、Redis 或 Docker 服务，也不需要 API 密钥。样例复用 MimirQ 已有的本地 BGE-M3 embedding provider、中文 BM25 分词器、RRF 融合评分和本地 BGE cross-encoder reranker；报告会保留文档来源和命中片段作为引用。

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

## 运行本地评测

项目后端环境需安装仓库依赖，以及本地模型所需的可选依赖：

```bash
pip install -r requirements-dev.txt
pip install sentence-transformers torch
python scripts/run_lab_hybrid_rag.py --out runs/lab_hybrid_rag.json
```

首次运行会从模型仓库下载 `BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3` 权重。若本机已经缓存权重，可离线运行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_lab_hybrid_rag.py
```

只评估 BGE-M3 向量检索、BM25 和 RRF 时可加 `--skip-reranker`。命令会打印融合排序与重排后排序的 Hit@5、MRR@5；JSON 报告包含逐题指标、模型名、检索配置和最终引用的 `chunk_id`、来源文件名及片段文本。Hit@5 表示前 5 条中至少命中一个标注相关片段的题目比例；MRR@5 是首个相关片段在前 5 条中的倒数排名的平均值。

该语料与查询仅用于链路冒烟和方法演示，评测数值只能描述这组合成样例，不代表真实实验室知识库或简历项目的质量结论。若要形成项目结果，应替换为经授权的领域语料和人工标注问题，并保存配置、语料版本与运行报告。

## 文件

- `config.json`：模型、RRF、重排、引用和指标参数，以及查询与 gold chunk 标注。
- `knowledge_base.json`：8 条合成实验室制度片段，包含 source metadata。
- `../../scripts/run_lab_hybrid_rag.py`：本地内存检索与评测入口。

## 来源与许可

样例脚本复用 MimirQ 仓库组件并随仓库的 Apache-2.0 许可分发；MimirQ 的第三方组件声明仍以根目录 `NOTICE` 为准。`knowledge_base.json` 是为本示例新写的合成内容，不包含真实单位制度或生产数据。
