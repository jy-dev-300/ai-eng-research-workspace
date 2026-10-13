# Adding an Adapter

1. Implement `TaskAdapter` in `harness/adapters/<domain>.py`.
2. Keep domain logic in its own repository; import its public API or invoke a documented command.
3. Return `AdapterRunOutput` with a harness result, domain output, measured values, and artifact payloads.
4. Return `AdapterEvaluation` with success, quality, typed metrics, extensible domain metrics, and evidence-backed failure subtype.
5. Register the factory in `register_builtin_adapters` (or register it at application startup).
6. Add a versioned evaluation fixture, YAML config, unit tests, and one dependency-light smoke test.

Do not catch errors merely to report success, use held-out answers to generate candidates, or label deterministic fixtures as official benchmark/model performance. Cleanup must be idempotent.

To add a harness, implement the existing `Harness` protocol and register a factory in `harness_registry`; `ExperimentRunner` requires no changes.

Build Agents use `ExperimentBuildAdapter`, which runs the approved validation commands and then invokes `ExperimentRunner` with the same domain config. A new domain therefore normally needs only its `TaskAdapter`, evaluation config, and build-adapter registry name—not a second evaluation implementation.
