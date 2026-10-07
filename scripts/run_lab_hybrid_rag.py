"""Run the synthetic, in-memory Hybrid RAG lab example without app services."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
# Importing MimirQ's app settings validates JWT configuration even though this
# standalone in-memory lab never serves requests. Use an ephemeral process-only
# key when the caller has not supplied one; never write it to disk or override
# an explicitly configured key.
os.environ.setdefault("SECRET_KEY", secrets.token_urlsafe(32))
# App module imports also construct the configured SQLAlchemy engine. Keep this
# standalone lab self-contained by defaulting to in-memory SQLite unless the
# caller explicitly selected another database. The lab itself never accesses it.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
# Keep the documented ``python scripts/run_lab_hybrid_rag.py`` invocation
# working from any current directory. Python otherwise adds only ``scripts/``
# to sys.path for a file-based entrypoint, so imports such as ``app.rag`` fail.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "examples" / "lab_hybrid_rag" / "config.json"
CONFIG_SCHEMA = "mimirq.lab_hybrid_rag.v1"
CORPUS_SCHEMA = "mimirq.lab_hybrid_rag_corpus.v1"


def load_experiment(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the experiment config and its synthetic corpus."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {CONFIG_SCHEMA}")

    runtime = config.get("runtime")
    retrieval = config.get("retrieval")
    if not isinstance(runtime, dict) or not isinstance(retrieval, dict):
        raise ValueError("config must define runtime and retrieval objects")
    if runtime.get("embedding_model_id") != "local/BAAI/bge-m3":
        raise ValueError("embedding_model_id must point to the supported local BGE-M3 provider")
    if runtime.get("fusion") != "rrf" or runtime.get("reranker_provider") != "local_bge_v2_m3":
        raise ValueError("this example expects RRF fusion and the local BGE v2-m3 reranker")
    for key in ("candidate_k", "rerank_top_n", "citation_top_n", "metric_k"):
        if int(retrieval.get(key, 0)) <= 0:
            raise ValueError(f"retrieval.{key} must be a positive integer")
    if int(retrieval["metric_k"]) < 5:
        raise ValueError("retrieval.metric_k must be at least 5 to report Hit@5 and MRR@5")

    corpus_path = (config_path.parent / str(config.get("corpus") or "")).resolve()
    if config_path.parent.resolve() not in corpus_path.parents:
        raise ValueError("corpus path must stay inside the example directory")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(corpus, dict) or corpus.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"corpus schema must be {CORPUS_SCHEMA}")
    documents = corpus.get("documents")
    queries = config.get("queries")
    if not isinstance(documents, list) or not documents:
        raise ValueError("corpus.documents must be a non-empty array")
    if not isinstance(queries, list) or not queries:
        raise ValueError("config.queries must be a non-empty array")

    chunk_ids: set[str] = set()
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"documents[{index}] must be an object")
        chunk_id = str(document.get("chunk_id") or "").strip()
        source = str(document.get("source") or "").strip()
        text = str(document.get("text") or "").strip()
        if not chunk_id or not source or not text:
            raise ValueError(f"documents[{index}] needs chunk_id, source, and text")
        if chunk_id in chunk_ids:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")
        chunk_ids.add(chunk_id)

    for index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f"queries[{index}] must be an object")
        if not str(query.get("question") or "").strip():
            raise ValueError(f"queries[{index}].question is required")
        relevant = query.get("relevant_chunk_ids")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"queries[{index}].relevant_chunk_ids must be non-empty")
        missing = [str(chunk_id) for chunk_id in relevant if str(chunk_id) not in chunk_ids]
        if missing:
            raise ValueError(f"queries[{index}] references unknown chunks: {missing}")
    return config, corpus


def rrf_rank(
    vector_ids: Sequence[str],
    bm25_ids: Sequence[str],
    *,
    rrf_k: int,
    score_function: Callable[..., float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse dense and BM25 rankings with the configured RRF score function."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    ranks_by_channel: list[dict[str, int]] = []
    for ranked_ids in (vector_ids, bm25_ids):
        ranks: dict[str, int] = {}
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            if chunk_id and chunk_id not in ranks:
                ranks[chunk_id] = rank
        ranks_by_channel.append(ranks)

    all_ids = set(ranks_by_channel[0]) | set(ranks_by_channel[1])
    rank_maps = {
        "vector": ranks_by_channel[0],
        "bm25": ranks_by_channel[1],
        "lexical": {},
        "sparse": {},
    }
    scores: list[tuple[str, float]] = []
    for chunk_id in all_ids:
        if score_function is None:
            score = sum(1.0 / (rrf_k + ranks[chunk_id]) for ranks in ranks_by_channel if chunk_id in ranks)
        else:
            score = float(score_function(rank_maps=rank_maps, key=chunk_id, k0=rrf_k))
        scores.append((chunk_id, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def retrieval_metrics(ranked_ids: Sequence[str], relevant_ids: Sequence[str], *, k: int = 5) -> dict[str, float]:
    """Calculate Hit@k and MRR@k for one query."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = {str(chunk_id) for chunk_id in relevant_ids if str(chunk_id)}
    top_ids = list(ranked_ids[:k])
    hit = any(chunk_id in relevant for chunk_id in top_ids)
    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(top_ids, start=1):
        if chunk_id in relevant:
            reciprocal_rank = 1.0 / rank
            break
    return {"hit_at_k": float(hit), "mrr_at_k": reciprocal_rank}


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must be non-empty and have matching dimensions")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Lab experiment JSON config")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "lab_hybrid_rag.json")
    parser.add_argument("--skip-reranker", action="store_true", help="Evaluate dense + BM25 + RRF only")
    parser.add_argument("--check", action="store_true", help="Validate config and corpus without loading models")
    return parser


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return round(sum(row[key] for row in rows) / len(rows), 6) if rows else 0.0


def _run(config: dict[str, Any], corpus: dict[str, Any], *, skip_reranker: bool) -> dict[str, Any]:
    # Heavy application and model dependencies stay out of config-check/test paths.
    from langchain_community.retrievers.bm25 import BM25Retriever
    from langchain_core.documents import Document

    from app.rag.embedding.factory import select_embedding_model
    from app.rag.preprocessing.tokenization import tokenize_for_bm25
    from app.rag.retrieval.hybrid.fusion import FusionMixin
    from app.rag.reranker.factory import get_reranker
    from app.rag.reranker.types import RerankCandidate

    runtime = config["runtime"]
    options = config["retrieval"]
    raw_documents = corpus["documents"]
    documents = [
        Document(
            page_content=str(row["text"]),
            id=str(row["chunk_id"]),
            metadata={"chunk_id": str(row["chunk_id"]), "source": str(row["source"])},
        )
        for row in raw_documents
    ]
    bm25 = BM25Retriever.from_documents(documents, preprocess_func=tokenize_for_bm25)
    bm25.k = min(int(options["candidate_k"]), len(documents))

    embedder = select_embedding_model(str(runtime["embedding_model_id"]))
    all_texts = [str(row["text"]) for row in raw_documents]
    all_texts.extend(str(query["question"]) for query in config["queries"])
    vectors = embedder.encode(all_texts)
    document_vectors = vectors[: len(raw_documents)]
    query_vectors = vectors[len(raw_documents) :]

    reranker = None
    if not skip_reranker:
        reranker = get_reranker(
            str(runtime["reranker_provider"]),
            model_name=str(runtime["reranker_model"]),
        )

    document_by_id = {str(document.metadata["chunk_id"]): document for document in documents}
    document_vector_by_id = {
        str(row["chunk_id"]): document_vectors[index]
        for index, row in enumerate(raw_documents)
    }
    fused_metrics: list[dict[str, float]] = []
    final_metrics: list[dict[str, float]] = []
    cases: list[dict[str, Any]] = []
    candidate_k = min(int(options["candidate_k"]), len(documents))
    metric_k = int(options["metric_k"])
    rrf_k = int(runtime["rrf_k"])

    for query_index, query in enumerate(config["queries"]):
        question = str(query["question"])
        dense_ids = sorted(
            document_by_id,
            key=lambda chunk_id: (
                -cosine_similarity(query_vectors[query_index], document_vector_by_id[chunk_id]),
                chunk_id,
            ),
        )[:candidate_k]
        bm25_documents = bm25.invoke(question)
        bm25_ids = [str(document.metadata["chunk_id"]) for document in bm25_documents]
        fused = rrf_rank(
            dense_ids,
            bm25_ids,
            rrf_k=rrf_k,
            score_function=FusionMixin._rrf_raw_score,
        )[:candidate_k]
        fused_ids = [chunk_id for chunk_id, _score in fused]
        fused_metrics.append(retrieval_metrics(fused_ids, query["relevant_chunk_ids"], k=metric_k))

        final_ids = fused_ids
        if reranker is not None and fused_ids:
            candidates = [
                RerankCandidate(
                    id=chunk_id,
                    text=document_by_id[chunk_id].page_content,
                    metadata=dict(document_by_id[chunk_id].metadata),
                )
                for chunk_id in fused_ids[: int(options["rerank_top_n"])]
            ]
            reranked = reranker.rerank(question, candidates, top_n=len(candidates))
            reranked_ids = [chunk_id for chunk_id in reranked.ordered_ids if chunk_id in document_by_id]
            seen = set(reranked_ids)
            final_ids = reranked_ids + [chunk_id for chunk_id in fused_ids if chunk_id not in seen]

        final_metrics.append(retrieval_metrics(final_ids, query["relevant_chunk_ids"], k=metric_k))
        citations = []
        for rank, chunk_id in enumerate(final_ids[: int(options["citation_top_n"])], start=1):
            document = document_by_id[chunk_id]
            citations.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "source": document.metadata["source"],
                    "text": document.page_content,
                }
            )
        cases.append(
            {
                "id": str(query.get("id") or ""),
                "question": question,
                "relevant_chunk_ids": list(query["relevant_chunk_ids"]),
                "dense_top_k": dense_ids,
                "bm25_top_k": bm25_ids,
                "rrf_top_k": fused_ids,
                "final_top_k": final_ids[:metric_k],
                "fused_metrics": fused_metrics[-1],
                "final_metrics": final_metrics[-1],
                "citations": citations,
            }
        )

    return {
        "schema": "mimirq.lab_hybrid_rag_report.v1",
        "models": {
            "embedding": str(runtime["embedding_model_id"]),
            "reranker": None if reranker is None else str(runtime["reranker_model"]),
        },
        "retrieval": {
            "dense_channel": "BGE-M3 cosine similarity",
            "sparse_channel": "BM25 with MimirQ Chinese tokenizer",
            "fusion": "RRF",
            "rrf_k": rrf_k,
            "metric_k": metric_k,
        },
        "summary": {
            "cases_total": len(cases),
            "fused_hit_at_5": _mean(fused_metrics, "hit_at_k"),
            "fused_mrr_at_5": _mean(fused_metrics, "mrr_at_k"),
            "final_hit_at_5": _mean(final_metrics, "hit_at_k"),
            "final_mrr_at_5": _mean(final_metrics, "mrr_at_k"),
            "reranker_enabled": reranker is not None,
        },
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config, corpus = load_experiment(args.config.expanduser().resolve())
        if args.check:
            print(
                "[lab-rag] CONFIG OK"
                f" documents={len(corpus['documents'])}"
                f" queries={len(config['queries'])}"
                f" embedding={config['runtime']['embedding_model_id']}"
                f" fusion={config['runtime']['fusion']}"
                f" reranker={config['runtime']['reranker_model']}"
            )
            return 0
        report = _run(config, corpus, skip_reranker=bool(args.skip_reranker))
        output_path = args.out.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary = report["summary"]
        print(
            "[lab-rag] OK"
            f" cases={summary['cases_total']}"
            f" fused_Hit@5={summary['fused_hit_at_5']:.4f}"
            f" fused_MRR@5={summary['fused_mrr_at_5']:.4f}"
            f" final_Hit@5={summary['final_hit_at_5']:.4f}"
            f" final_MRR@5={summary['final_mrr_at_5']:.4f}"
        )
        print(f"[lab-rag] report={output_path}")
        return 0
    except Exception as exc:
        print(f"[lab-rag] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
