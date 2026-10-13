from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from harness.models import ExperimentConfig, Metric
from harnesses.contracts import Harness, HarnessResult


@dataclass
class AdapterRunOutput:
    value: dict[str, Any] = field(default_factory=dict)
    harness_result: HarnessResult | None = None
    metrics: dict[str, float | int | bool | None] = field(default_factory=dict)
    artifact_payloads: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterEvaluation:
    success: bool
    quality: float | None = None
    metrics: list[Metric] = field(default_factory=list)
    domain_metrics: dict[str, float | int | bool | None] = field(default_factory=dict)
    failure_subtype: str | None = None
    evidence: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class TaskAdapter(ABC):
    """Owns domain behavior while the runner owns orchestration."""

    @abstractmethod
    def load_tasks(self, config: ExperimentConfig) -> list[dict[str, Any]]: ...

    @abstractmethod
    def setup(self, task: dict[str, Any], config: ExperimentConfig) -> None: ...

    @abstractmethod
    async def run(
        self, task: dict[str, Any], harness: Harness, config: ExperimentConfig
    ) -> AdapterRunOutput: ...

    @abstractmethod
    def evaluate(
        self, task: dict[str, Any], run_output: AdapterRunOutput, config: ExperimentConfig
    ) -> AdapterEvaluation: ...

    def artifacts(
        self,
        task: dict[str, Any],
        run_output: AdapterRunOutput,
        evaluation: AdapterEvaluation,
    ) -> dict[str, Any]:
        return run_output.artifact_payloads

    def classify_failure(
        self,
        task: dict[str, Any],
        run_output: AdapterRunOutput,
        evaluation: AdapterEvaluation,
    ) -> tuple[str | None, list[str]]:
        return evaluation.failure_subtype, evaluation.evidence

    def cleanup(self) -> None:
        return None
