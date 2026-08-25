"""Deterministic end-to-end coverage that never contacts an LLM provider."""

import pytest

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How do I reset my password?", Route.SIMPLE),
        ("Look up order status 42", Route.TOOL),
        ("Can you fix it?", Route.MISSING_INFO),
        ("Delete this account", Route.RISKY),
        ("Service timeout failure", Route.ERROR),
    ],
)
def test_all_routes_terminate_offline(
    monkeypatch: pytest.MonkeyPatch, query: str, expected: Route
) -> None:
    monkeypatch.setenv("LLM_OFFLINE", "true")
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "false")
    graph = build_graph(build_checkpointer("memory"))
    scenario = Scenario(id=expected.value, query=query, expected_route=expected)
    state = initial_state(scenario)

    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    assert result["route"] == expected.value
    assert result.get("final_answer") or result.get("pending_question")
    assert result["events"][-1]["node"] == "finalize"
    if expected is Route.RISKY:
        assert result["approval"]["approved"] is True


def test_sqlite_checkpointer_retains_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_OFFLINE", "true")
    checkpointer = build_checkpointer("sqlite", ":memory:")
    graph = build_graph(checkpointer)
    scenario = Scenario(id="sqlite", query="Password reset help", expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}

    result = graph.invoke(state, config=config)

    assert result["events"][-1]["node"] == "finalize"
    assert len(list(graph.get_state_history(config))) >= 2
