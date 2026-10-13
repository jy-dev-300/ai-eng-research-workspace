from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class TraceEventView(BaseModel):
    sequence: int
    type: str
    payload: dict[str, Any]


class DemoTaskResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    task_id: str
    status: str
    expected_answer: str
    final_answer: str | None
    correct: bool
    model_calls: int
    tool_calls: int
    trace: list[TraceEventView]


class DemoRunResponse(BaseModel):
    backend: str
    warning: str
    tasks: list[DemoTaskResult]
