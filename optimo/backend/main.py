from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import DemoRunResponse, DemoTaskResult, TraceEventView
from benchmarks.deterministic import DETERMINISTIC_SUITE
from harnesses.simple_tool_loop import SimpleToolLoop
from serving.mock import DeterministicMockBackend
from tools.registry import default_tool_registry
from tracing.recorder import TraceRecorder

app = FastAPI(title="Optimo API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/benchmarks")
async def benchmarks() -> dict:
    return DETERMINISTIC_SUITE.model_dump()


@app.post("/api/runs/demo", response_model=DemoRunResponse)
async def run_demo() -> DemoRunResponse:
    backend = DeterministicMockBackend()
    tools = default_tool_registry()
    task_results: list[DemoTaskResult] = []
    for task in DETERMINISTIC_SUITE.tasks:
        runner = SimpleToolLoop(
            backend,
            tools,
            TraceRecorder(str(uuid4()), task.id),
        )
        result = await runner.run(task)
        task_results.append(
            DemoTaskResult(
                task_id=task.id,
                status=result.status.value,
                expected_answer=task.expected_answer,
                final_answer=result.final_answer,
                correct=result.final_answer == task.expected_answer,
                model_calls=result.model_calls,
                tool_calls=result.tool_calls,
                trace=[
                    TraceEventView(
                        sequence=event.sequence,
                        type=event.type.value,
                        payload=event.payload,
                    )
                    for event in result.trace
                ],
            )
        )
    return DemoRunResponse(
        backend=backend.name,
        warning="Mock metrics prove deterministic behavior, not inference performance.",
        tasks=task_results,
    )
