# AI Engineering Research Workspace

This repository is the workspace-level entry point for the AI Systems Research Lab. The Lab lives
in `optimo/` and coordinates independent systems through adapters rather than copying their
implementations into the harness.

```text
program.md
  -> workflow classification
  -> Build / Eval / Research loop
  -> project adapter
  -> independent system
  -> ExperimentResult / MLflow / exact diff
```

## Repositories

- `optimo/` — Lab orchestration, execution harnesses, Build Loop, ExperimentRunner, Research Brain,
  program runner, adapters, tests, and documentation.
- `bo/` — independent retrieval/RAG system, pinned as a Git submodule.
- `myvoice/` — local mount point for the independent personalized-generation system.
- `swe-bench/` — local mount point for the independent SWE-bench checkout.
- `ManiSkill/` — local mount point for the independent embodied-AI checkout.

The last three directories may be populated locally without being vendored into this repository.
Their locations and runtime dependencies are resolved by the Lab's project registry and adapters.

## Clone

```powershell
git clone --recurse-submodules https://github.com/jy-dev-300/ai-eng-research-workspace.git
cd ai-eng-research-workspace\optimo
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m pytest -q
```

See `optimo/README.md` and `optimo/docs/PROGRAMS.md` for architecture and program-runner usage.
