import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_lab_hybrid_rag import load_experiment, retrieval_metrics, rrf_rank


class LabHybridRagTests(unittest.TestCase):
    def test_cli_check_works_outside_repository_without_runtime_dependencies(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "scripts" / "run_lab_hybrid_rag.py"

        with TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [sys.executable, str(script), "--check"],
                cwd=temp_dir,
                env={**os.environ, "PYTHONPATH": ""},
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[lab-rag] CONFIG OK", result.stdout)

    def test_rrf_rewards_chunks_present_in_both_channels(self) -> None:
        ranked = rrf_rank(["dense", "shared"], ["shared", "bm25"], rrf_k=60)

        self.assertEqual(ranked[0][0], "shared")
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_hit_at_five_and_mrr_at_five_use_first_relevant_rank(self) -> None:
        metrics = retrieval_metrics(["wrong", "gold", "other"], ["gold"], k=5)

        self.assertEqual(metrics, {"hit_at_k": 1.0, "mrr_at_k": 0.5})

    def test_unretrieved_relevance_has_zero_metrics(self) -> None:
        metrics = retrieval_metrics(["a", "b"], ["gold"], k=5)

        self.assertEqual(metrics, {"hit_at_k": 0.0, "mrr_at_k": 0.0})

    def test_sample_config_and_corpus_have_valid_citation_targets(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "examples" / "lab_hybrid_rag" / "config.json"

        config, corpus = load_experiment(config_path)

        self.assertEqual(len(corpus["documents"]), 8)
        self.assertEqual(len(config["queries"]), 4)

    def test_unknown_gold_chunk_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            corpus_path = directory / "corpus.json"
            config_path = directory / "config.json"
            corpus_path.write_text(
                json.dumps(
                    {
                        "schema": "mimirq.lab_hybrid_rag_corpus.v1",
                        "documents": [{"chunk_id": "known", "source": "doc.md", "text": "text"}],
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "schema": "mimirq.lab_hybrid_rag.v1",
                        "runtime": {
                            "embedding_model_id": "local/BAAI/bge-m3",
                            "fusion": "rrf",
                            "reranker_provider": "local_bge_v2_m3",
                        },
                        "retrieval": {
                            "candidate_k": 5,
                            "rerank_top_n": 5,
                            "citation_top_n": 3,
                            "metric_k": 5,
                        },
                        "corpus": "corpus.json",
                        "queries": [{"question": "q", "relevant_chunk_ids": ["missing"]}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown chunks"):
                load_experiment(config_path)


if __name__ == "__main__":
    unittest.main()
