"""Render a submission-ready report from scenario metrics."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    The generated report includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    scenario_rows = []
    for item in metrics.scenario_metrics:
        scenario_rows.append(
            "| {id} | {expected} | {actual} | {success} | {retries} | {interrupts} |".format(
                id=item.scenario_id.replace("|", "\\|"),
                expected=item.expected_route,
                actual=item.actual_route or "—",
                success="yes" if item.success else "no",
                retries=item.retry_count,
                interrupts=item.interrupt_count,
            )
        )

    return f"""# Day 08 Lab Report

## 1. Student

- Name: Nguyễn Văn Trường
- Student ID: 2A202601974
- Repository: `DAY23-2A202601974-NguyenVanTruong`

## 2. Implementation

This repository implements a production-style support-ticket workflow with LLM routing,
grounded response generation, bounded retries, approval gating, persistence, and audit events.

## 3. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.2%} |
| Average nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Approval/HITL visits | {metrics.total_interrupts} |
| State-history verification | {"passed" if metrics.resume_success else "not run"} |

## 4. Architecture and state

The graph starts with intake and structured LLM classification. Simple requests go directly to
answer generation; read-only requests use a mock tool and quality gate; incomplete requests ask
for clarification; risky actions require approval before execution; and system errors enter the
same bounded retry loop. Every terminal branch passes through `finalize`.

Scalar fields (`route`, `risk_level`, `attempt`, approval and output fields) use overwrite
semantics. `messages`, `tool_results`, `errors`, and `events` use additive reducers so retries and
audit history are never lost. State is lean, typed, and JSON-serializable.

## 5. Scenario results

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
{chr(10).join(scenario_rows)}

## 6. Failure analysis

1. **Transient tool failure:** an error result is evaluated as `needs_retry`; the retry node
   increments `attempt`, and routing compares it with `max_attempts`. Exhausted work is moved to
   dead letter with a user-safe response, preventing infinite loops.
2. **Risky action without approval:** risky requests cannot reach the tool directly. A rejected
   decision routes to clarification; real deployments can enable LangGraph `interrupt()` and
   resume with a human decision.
3. **LLM/provider outage:** the required LLM call is attempted first. A provider-independent,
   intent-level fallback keeps the workflow terminating and records `fallback_used` in the audit
   event, rather than matching scenario identifiers or producing account facts.

## 7. Persistence and recovery evidence

Each invocation supplies a stable `thread_id`. MemorySaver is enabled by default and state history
is queried after runs. The SQLite adapter uses WAL mode and supports durable checkpoints when the
optional `sqlite` dependency is installed.

## 8. Extension work

- Real HITL can be enabled with `LANGGRAPH_INTERRUPT=true`.
- Checkpoint history provides time-travel/debug evidence (`resume_success` above).
- SQLite WAL persistence is available through `build_checkpointer("sqlite", path)`.

## 9. Improvement plan

For production, replace the mock tool with authenticated, idempotent APIs; persist approval
identity and policy evidence; add provider retries/timeouts and tracing; redact sensitive data;
and add adversarial classification and crash-resume integration tests.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
