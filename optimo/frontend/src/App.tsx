import { useState } from 'react'

type TraceEvent = {
  sequence: number
  type: string
  payload: Record<string, unknown>
}

type TaskResult = {
  task_id: string
  status: string
  expected_answer: string
  final_answer: string | null
  correct: boolean
  model_calls: number
  tool_calls: number
  trace: TraceEvent[]
}

type DemoRun = {
  backend: string
  warning: string
  tasks: TaskResult[]
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export function App() {
  const [run, setRun] = useState<DemoRun | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function launchDemo() {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_URL}/api/runs/demo`, { method: 'POST' })
      if (!response.ok) throw new Error(`API returned ${response.status}`)
      const data: DemoRun = await response.json()
      setRun(data)
      setSelected(data.tasks[0]?.task_id ?? null)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unknown request error')
    } finally {
      setLoading(false)
    }
  }

  const selectedTask = run?.tasks.find((task) => task.task_id === selected)

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">Agent Harness + Inference Optimizer</p>
          <h1>Make agent behavior measurable.</h1>
          <p className="lede">Compare control loops and serving configurations without confusing task quality with GPU performance.</p>
        </div>
        <button onClick={launchDemo} disabled={loading}>{loading ? 'Running…' : 'Run deterministic baseline'}</button>
      </header>

      <section className="boundary" aria-label="Architecture layers">
        <article><span>Layer 1</span><h2>Harness quality</h2><p>Tools, state, retries, termination, and task success.</p></article>
        <div className="arrow" aria-hidden="true">→</div>
        <article><span>Layer 2</span><h2>Inference performance</h2><p>Queueing, caching, generation, latency, and throughput.</p></article>
      </section>

      {error && <p className="error" role="alert">{error}</p>}
      {!run && <section className="empty"><p>Run the mock baseline to inspect a complete model → tool → model trajectory.</p><small>It proves deterministic wiring, not real inference performance.</small></section>}

      {run && <>
        <section className="notice"><strong>{run.backend}</strong><span>{run.warning}</span></section>
        <section className="workspace">
          <div className="task-list">
            <h2>Benchmark tasks</h2>
            {run.tasks.map((task) => <button key={task.task_id} className={selected === task.task_id ? 'active task' : 'task'} onClick={() => setSelected(task.task_id)}>
              <span>{task.task_id}</span><b>{task.correct ? 'pass' : 'fail'}</b><small>{task.model_calls} model · {task.tool_calls} tool</small>
            </button>)}
          </div>
          <div className="trace">
            <h2>Trajectory</h2>
            <p>Expected <code>{selectedTask?.expected_answer}</code> · received <code>{selectedTask?.final_answer}</code></p>
            {selectedTask?.trace.map((event) => <details key={event.sequence} open={event.type === 'tool_call'}>
              <summary><span>{String(event.sequence + 1).padStart(2, '0')}</span>{event.type.replace('_', ' ')}</summary>
              <pre>{JSON.stringify(event.payload, null, 2)}</pre>
            </details>)}
          </div>
        </section>
      </>}
    </main>
  )
}

