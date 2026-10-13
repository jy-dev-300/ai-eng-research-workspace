from __future__ import annotations

from typing import Any

from harness.adapters.base import AdapterEvaluation, AdapterRunOutput, TaskAdapter
from harness.adapters.common import load_json_records, resolve_workspace_path, run_reasoning_harness
from harness.models import ExperimentConfig, Metric
from harnesses.contracts import Harness, RunStatus


class HarnessAdapter(TaskAdapter):
    """Makes the Lab's execution harnesses measurable through their own experiment engine."""

    def load_tasks(self, config: ExperimentConfig) -> list[dict[str, Any]]:
        dataset = config.evaluation.dataset or "evals/harness_smoke_v1.json"
        return load_json_records(resolve_workspace_path(config, dataset))

    def setup(self, task: dict[str, Any], config: ExperimentConfig) -> None:
        if "expected_answer" not in task:
            raise ValueError("Harness task requires expected_answer")

    async def run(
        self, task: dict[str, Any], harness: Harness, config: ExperimentConfig
    ) -> AdapterRunOutput:
        result = await run_reasoning_harness(
            harness,
            task_id=str(task["id"]),
            prompt=str(task["prompt"]),
            expected_answer=str(task["expected_answer"]),
        )
        return AdapterRunOutput(
            value={"answer": result.final_answer or ""},
            harness_result=result,
            artifact_payloads={"agent_output.txt": result.final_answer or ""},
        )

    def evaluate(
        self, task: dict[str, Any], run_output: AdapterRunOutput, config: ExperimentConfig
    ) -> AdapterEvaluation:
        expected = str(task["expected_answer"]).strip().casefold()
        actual = str(run_output.value["answer"]).strip().casefold()
        succeeded = actual == expected and (
            run_output.harness_result is not None
            and run_output.harness_result.status is RunStatus.SUCCEEDED
        )
        quality = 1.0 if succeeded else 0.0
        return AdapterEvaluation(
            success=succeeded,
            quality=quality,
            metrics=[Metric(name="task_success", value=quality)],
            domain_metrics={"task_success": succeeded},
            failure_subtype=None if succeeded else "harness_answer_mismatch",
            evidence=[] if succeeded else [f"expected={expected!r}, actual={actual!r}"],
        )
