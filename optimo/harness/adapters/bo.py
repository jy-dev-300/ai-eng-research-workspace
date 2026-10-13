from __future__ import annotations

import math
import sys
from statistics import mean
from time import perf_counter
from typing import Any

from harness.adapters.base import AdapterEvaluation, AdapterRunOutput, TaskAdapter
from harness.adapters.common import load_json_records, resolve_workspace_path, run_reasoning_harness
from harness.models import ExperimentConfig, Metric
from harnesses.contracts import Harness


class BoAdapter(TaskAdapter):
    """Adapter over Bo's public retrieval modules; Bo remains a sibling system."""

    def __init__(self) -> None:
        self._chunks: list[Any] = []
        self._retriever: Any = None

    def load_tasks(self, config: ExperimentConfig) -> list[dict[str, Any]]:
        dataset = config.evaluation.dataset or "../bo/sample_corpus/retrieval_gold.jsonl"
        return load_json_records(resolve_workspace_path(config, dataset))

    def setup(self, task: dict[str, Any], config: ExperimentConfig) -> None:
        bo_root = resolve_workspace_path(
            config, str(config.adapter.parameters.get("repo", "../bo"))
        )
        if not bo_root.exists():
            raise FileNotFoundError(f"Bo repository not found: {bo_root}")
        if str(bo_root) not in sys.path:
            sys.path.insert(0, str(bo_root))
        from retrieval.corpus import load_chunks_jsonl
        from retrieval.lexical import BM25Index

        corpus = config.adapter.parameters.get("corpus", "sample_corpus/chunks.jsonl")
        self._chunks = load_chunks_jsonl((bo_root / str(corpus)).resolve())
        stage = str(config.parameters.get("retrieval", {}).get("stage", "bm25"))
        if stage == "bm25":
            self._retriever = BM25Index(self._chunks)
        else:
            # This calls Bo's real BGE/hybrid/reranking implementation. Missing
            # model assets surface as explicit resource failures in the runner.
            from retrieval.stages import build_retriever

            self._retriever = build_retriever(self._chunks, stage)  # type: ignore[arg-type]

    async def run(
        self, task: dict[str, Any], harness: Harness, config: ExperimentConfig
    ) -> AdapterRunOutput:
        retrieval = config.parameters.get("retrieval", {})
        top_k = int(retrieval.get("top_k", 5))
        started = perf_counter()
        candidates = self._retriever.search(str(task["query"]), limit=top_k)
        retrieval_ms = (perf_counter() - started) * 1_000
        rankings = [
            {
                "chunk_id": item.chunk.id,
                "rank": item.rank,
                "score": item.score,
                "source": item.source,
                "text": item.chunk.text,
            }
            for item in candidates
        ]
        answer = rankings[0]["text"] if rankings else "No evidence retrieved."
        harness_result = await run_reasoning_harness(
            harness,
            task_id=str(task["id"]),
            prompt=f"Use this retrieved evidence to answer the query: {task['query']}\n{answer}",
            expected_answer=answer,
        )
        return AdapterRunOutput(
            value={"query": task["query"], "rankings": rankings, "answer": answer},
            harness_result=harness_result,
            metrics={"retrieval_latency_ms": retrieval_ms},
            artifact_payloads={
                "retrieval.json": {"query": task["query"], "rankings": rankings},
                "agent_output.txt": harness_result.final_answer or "",
            },
        )

    def evaluate(
        self, task: dict[str, Any], run_output: AdapterRunOutput, config: ExperimentConfig
    ) -> AdapterEvaluation:
        retrieved = [item["chunk_id"] for item in run_output.value["rankings"]]
        relevant = set(task["relevant_chunk_ids"])
        top_k = int(config.parameters.get("retrieval", {}).get("top_k", 5))
        hits = len(set(retrieved[:top_k]) & relevant)
        recall = hits / len(relevant)
        first_rank = next((i for i, item in enumerate(retrieved, 1) if item in relevant), None)
        mrr = 0.0 if first_rank is None else 1.0 / first_rank
        dcg = sum(
            1.0 / math.log2(rank + 1) for rank, item in enumerate(retrieved, 1) if item in relevant
        )
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), top_k) + 1))
        ndcg = dcg / ideal if ideal else 0.0
        subtype = None if recall == 1 else "relevant_chunk_not_retrieved"
        latency = float(run_output.metrics.get("retrieval_latency_ms") or 0)
        return AdapterEvaluation(
            success=recall == 1.0,
            quality=mean([recall, mrr, ndcg]),
            metrics=[
                Metric(name=f"recall_at_{top_k}", value=recall),
                Metric(name="mrr", value=mrr),
                Metric(name="ndcg", value=ndcg),
                Metric(name="retrieval_latency_ms", value=latency, unit="ms"),
            ],
            domain_metrics={"recall_at_k": recall, "mrr": mrr, "ndcg": ndcg},
            failure_subtype=subtype,
            evidence=[]
            if subtype is None
            else [f"relevant={sorted(relevant)}, retrieved={retrieved}"],
            details={"relevant_chunk_ids": sorted(relevant), "retrieved_chunk_ids": retrieved},
        )

    def cleanup(self) -> None:
        self._chunks = []
        self._retriever = None
