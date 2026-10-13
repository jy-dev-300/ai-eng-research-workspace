from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from harness.adapters.base import AdapterEvaluation, AdapterRunOutput, TaskAdapter
from harness.adapters.common import load_json_records, resolve_workspace_path, run_reasoning_harness
from harness.models import ExperimentConfig, Metric
from harnesses.contracts import Harness


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _style_similarity(candidate: str, profile: list[str]) -> float:
    profile_tokens = set(_tokens(" ".join(profile)))
    candidate_tokens = set(_tokens(candidate))
    return len(candidate_tokens & profile_tokens) / max(1, len(candidate_tokens))


class MyVoiceAdapter(TaskAdapter):
    """Held-out completion evaluator with base, prompt, and retrieval strategies."""

    def load_tasks(self, config: ExperimentConfig) -> list[dict[str, Any]]:
        dataset = config.evaluation.dataset or "evals/myvoice_heldout_v1.json"
        return load_json_records(resolve_workspace_path(config, dataset))

    def setup(self, task: dict[str, Any], config: ExperimentConfig) -> None:
        profile = set(task.get("profile_examples", []))
        if task.get("reference") in profile:
            raise ValueError(
                f"Held-out reference leaked into profile examples for {task.get('id')}"
            )

    def _generate(self, task: dict[str, Any], strategy: str) -> str:
        prompt = str(task["prompt"])
        profile = list(task.get("profile_examples", []))
        if strategy == "base":
            return "Thank you for reaching out. I would be happy to discuss this further."
        if not profile:
            return "Thanks — I will take a look and follow up."
        # Local deterministic personalization uses only profile data, never the held-out reference.
        selected = max(profile, key=lambda text: SequenceMatcher(None, prompt, text).ratio())
        if strategy == "prompt_personalization":
            return selected.split(".")[0].strip() + "."
        if strategy == "retrieval_personalization":
            words = selected.split()
            return " ".join(words[-min(12, len(words)) :]).strip()
        raise ValueError(f"Unknown MyVoice strategy: {strategy}")

    async def run(
        self, task: dict[str, Any], harness: Harness, config: ExperimentConfig
    ) -> AdapterRunOutput:
        strategy = str(config.parameters.get("personalization", {}).get("strategy", "base"))
        completion = self._generate(task, strategy)
        harness_result = await run_reasoning_harness(
            harness,
            task_id=str(task["id"]),
            prompt=f"Complete in the user's style: {task['prompt']}",
            expected_answer=completion,
        )
        return AdapterRunOutput(
            value={"prompt": task["prompt"], "completion": completion, "strategy": strategy},
            harness_result=harness_result,
            artifact_payloads={
                "completion.json": {
                    "prompt": task["prompt"],
                    "completion": completion,
                    "reference": task["reference"],
                    "strategy": strategy,
                }
            },
        )

    def evaluate(
        self, task: dict[str, Any], run_output: AdapterRunOutput, config: ExperimentConfig
    ) -> AdapterEvaluation:
        candidate = str(run_output.value["completion"])
        reference = str(task["reference"])
        quality = SequenceMatcher(None, candidate.casefold(), reference.casefold()).ratio()
        style = _style_similarity(candidate, list(task.get("profile_examples", [])))
        acceptance = (
            1.0
            if quality >= float(config.evaluation.parameters.get("acceptance_threshold", 0.45))
            else 0.0
        )
        saved = max(0, min(len(candidate), round(len(reference) * quality)) - 1)
        metrics = {
            "completion_quality": quality,
            "acceptance_proxy": acceptance,
            "keystrokes_saved": saved,
            "style_similarity": style,
        }
        return AdapterEvaluation(
            success=bool(acceptance),
            quality=quality,
            metrics=[Metric(name=name, value=float(value)) for name, value in metrics.items()],
            domain_metrics=metrics,
            failure_subtype=None if acceptance else "completion_rejected_by_proxy",
            evidence=[] if acceptance else [f"quality={quality:.3f} below threshold"],
        )
