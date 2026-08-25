"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


class RouteClassification(BaseModel):
    """Strict response schema used by the classifier LLM."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    reasoning: str = Field(description="A short reason for selecting this route")


def _content_as_text(response: object) -> str:
    """Normalize LangChain provider responses without depending on one provider."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in content]
        return " ".join(part for part in parts if part).strip()
    return str(content).strip()


def _fallback_route(query: str) -> str:
    """Provider-failure fallback; categories are intent based, never scenario based."""
    text = query.casefold()
    patterns = {
        "risky": r"\b(refund|delete|remove|cancel|terminate|send|email|charge|pay|update)\b",
        "tool": r"\b(lookup|look up|find|search|status|track|tracking|order|invoice)\b",
        "missing_info": r"^(can you |please )?(fix|help|solve|do)\s+(it|this|that)\??$",
        "error": r"\b(error|failure|failed|timeout|crash|unavailable|exception|cannot recover)\b",
    }
    for route in ("risky", "tool", "missing_info", "error"):
        if re.search(patterns[route], text):
            return route
    return "simple"


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Workflow nodes ──────────────────────────────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    prompt = f"""You route customer-support tickets. Classify the ticket into exactly one route.
Classify the user's requested mode of help, not isolated action words. A how-to question asking
for instructions the user can follow is simple, even when those instructions concern a password
reset or another account operation. Use risky only when the user asks the support agent/system to
perform the side effect on their behalf.

Apply this priority when more than one route genuinely fits:
1. risky: asks us to execute a side effect (refund, delete, cancel, modify data, send messages)
2. tool: a read-only lookup, search, status or tracking request
3. missing_info: too vague or incomplete to act on
4. error: reports a timeout, crash, outage, failure, or service error
5. simple: general or self-service guidance answerable without a tool

Ticket: {query!r}
Return the route and a short reason."""
    fallback_used = False
    reason = "classified by LLM"
    route: str
    try:
        classifier = get_llm(temperature=0.0).with_structured_output(RouteClassification)
        result = classifier.invoke(prompt)
        parsed = (
            result
            if isinstance(result, RouteClassification)
            else RouteClassification.model_validate(result)
        )
        route = parsed.route
        reason = parsed.reasoning
    except Exception as exc:
        route = _fallback_route(query)
        fallback_used = True
        reason = f"LLM unavailable; intent fallback used ({type(exc).__name__})"
    return {
        "route": route,
        "risk_level": "high" if route == "risky" else "low",
        "events": [
            make_event(
                "classify",
                "completed",
                reason,
                route=route,
                fallback_used=fallback_used,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0))
    route = state.get("route", "")
    if route == "error" and attempt < 2:
        result = f"ERROR: transient support service failure on attempt {attempt + 1}"
        event_type = "failed"
    elif route == "risky":
        proposed = state.get("proposed_action", state.get("query", ""))
        result = f"Action completed after approval: {proposed}"
        event_type = "completed"
    else:
        result = f"Mock support lookup completed for: {state.get('query', '')}"
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, result, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    results = state.get("tool_results", [])
    latest = results[-1] if results else "ERROR: tool returned no result"
    evaluation = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": evaluation,
        "events": [make_event("evaluate", "completed", f"tool result: {evaluation}")],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    context = "\n".join(tool_results) if tool_results else "No tool was required."
    prompt = f"""You are a concise customer-support agent. Answer the ticket using only the
provided context. Do not invent account-specific facts. If an approved action was completed,
confirm only what the context says.

Ticket: {query}
Tool context: {context}
Approval: {approval or 'not applicable'}

Write a helpful final response."""
    fallback_used = False
    try:
        answer = _content_as_text(get_llm(temperature=0.1).invoke(prompt))
        if not answer:
            raise ValueError("LLM returned an empty answer")
    except Exception:
        fallback_used = True
        if tool_results:
            answer = f"Your request was processed. Result: {tool_results[-1]}"
        else:
            answer = (
                "To resolve this request, follow your organization's standard support procedure "
                "or contact an administrator if you cannot complete the steps."
            )
    return {
        "final_answer": answer,
        "messages": [f"assistant:{answer}"],
        "events": [
            make_event(
                "answer",
                "completed",
                "grounded response generated",
                fallback_used=fallback_used,
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    approval = state.get("approval")
    if approval and not approval.get("approved"):
        question = "The proposed action was not approved. What safer alternative would you like?"
    else:
        question = (
            f"Could you provide the affected product or account, the expected outcome, and any "
            f"error details for your request ({query!r})?"
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": [f"assistant:{question}"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    proposed = f"Execute requested side-effecting support action: {state.get('query', '').strip()}"
    return {
        "proposed_action": proposed,
        "events": [make_event("risky_action", "approval_required", proposed, risk_level="high")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return approval decision and an audit event.
    """
    interrupt_enabled = os.getenv("LANGGRAPH_INTERRUPT", "false").casefold() in {
        "1",
        "true",
        "yes",
    }
    batch_mode = os.getenv("LANGGRAPH_BATCH_MODE", "false").casefold() in {"1", "true", "yes"}
    if interrupt_enabled and not batch_mode:
        from langgraph.types import interrupt

        response = interrupt(
            {
                "question": "Approve this risky support action?",
                "proposed_action": state.get("proposed_action", ""),
            }
        )
        if isinstance(response, bool):
            decision = ApprovalDecision(approved=response, reviewer="human")
        else:
            decision = ApprovalDecision.model_validate(response)
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Automatically approved for deterministic lab/CI execution.",
        )
    payload = decision.model_dump()
    return {
        "approval": payload,
        "events": [make_event("approval", "completed", "approval decision recorded", **payload)],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0)) + 1
    error = f"Transient tool failure recorded; retry attempt {attempt}"
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [make_event("retry", "scheduled", error, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0))
    answer = (
        "We could not complete this request after the configured retry limit. "
        "It has been escalated for manual investigation."
    )
    return {
        "final_answer": answer,
        "events": [make_event("dead_letter", "escalated", answer, attempts=attempt)],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
