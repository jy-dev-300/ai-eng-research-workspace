from __future__ import annotations

import re
from pathlib import Path

from harness.programs.models import ProgramDefinition, ProgramDocument


class ProgramRegistry:
    def __init__(self, lab_root: str | Path | None = None) -> None:
        self.lab_root = Path(lab_root or Path.cwd()).resolve()
        common = ("inspect", "build", "test", "evaluate", "experiment", "research")
        self._projects = {
            "bo": ProgramDefinition(
                project="Bo",
                adapter="bo",
                repo="../bo",
                capabilities=common,
                state_summary="BM25, dense, hybrid RRF, BGE/JEV reranking, and retrieval evals.",
                evaluation_config="configs/bo/hybrid.yaml",
                aliases=("bo",),
            ),
            "myvoice": ProgramDefinition(
                project="MyVoice",
                adapter="myvoice",
                repo="../myvoice",
                capabilities=common,
                state_summary=(
                    "Held-out completion evals with base, prompt, and retrieval personalization."
                ),
                evaluation_config="configs/myvoice/retrieval.yaml",
                aliases=("myvoice", "my voice"),
            ),
            "swe": ProgramDefinition(
                project="SWE-bench",
                adapter="swebench",
                repo="../swe-bench",
                capabilities=common,
                state_summary=(
                    "Local coding-agent smoke tasks and an official SWE-bench integration boundary."
                ),
                evaluation_config="configs/swebench/simple_loop.yaml",
                aliases=("swe", "swe-bench", "swebench"),
            ),
            "maniskill": ProgramDefinition(
                project="ManiSkill",
                adapter="maniskill",
                repo="../ManiSkill",
                capabilities=common,
                state_summary=(
                    "Headless deterministic smoke loop and optional real PickCube-v1 episode."
                ),
                evaluation_config="configs/maniskill/reach_smoke.yaml",
                aliases=("maniskill", "physical-ai", "physical ai"),
            ),
            "harness": ProgramDefinition(
                project="Harness",
                adapter="harness",
                repo=".",
                capabilities=common,
                state_summary=(
                    "Simple loop, planner/executor, retries, tracing, profiling, "
                    "and A-M experiments."
                ),
                evaluation_config="configs/harness/baseline.yaml",
                aliases=("harness", "optimo"),
            ),
        }

    def resolve(self, program: ProgramDocument) -> ProgramDefinition:
        stem = Path(program.filename).stem.casefold()
        stem = stem.removesuffix("-program")
        candidates = [key for key in self._projects if stem == key or stem.startswith(f"{key}-")]
        heading = re.search(r"^#\s+(.+?)\s*$", program.contents, flags=re.MULTILINE)
        if not candidates and heading:
            title = heading.group(1).casefold()
            candidates = [
                key
                for key, value in self._projects.items()
                if title == key or title in value.aliases
            ]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot resolve one project for {program.filename!r}; use a registered project "
                "prefix or matching Markdown H1"
            )
        return self._projects[candidates[0]]

    def repo_path(self, definition: ProgramDefinition) -> Path:
        return (self.lab_root / definition.repo).resolve()

    def config_path(self, definition: ProgramDefinition) -> Path:
        return (self.lab_root / definition.evaluation_config).resolve()

    def definitions(self) -> tuple[ProgramDefinition, ...]:
        return tuple(self._projects.values())
