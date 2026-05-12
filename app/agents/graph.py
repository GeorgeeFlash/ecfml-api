from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, TimeoutPolicy

from app.agents.nodes.data_preparation import data_preparation_node
from app.agents.nodes.forecasting import forecasting_node
from app.agents.nodes.revision import revision_node
from app.agents.nodes.validation import validation_node
from app.agents.state import AgentState

MAX_REVISIONS = 2


def should_revise(state: AgentState) -> str:
    report: dict = state.get("validation_report") or {}
    if report.get("anomaly_pct", 0) > 10 and state.get("revision_count", 0) < MAX_REVISIONS:
        return "revision"
    return END


def compile_graph():
    builder = StateGraph(AgentState)

    builder.add_node("data_preparation", data_preparation_node)
    builder.add_node(
        "forecasting",
        forecasting_node,
        retry_policy=RetryPolicy(max_attempts=3),
        timeout=TimeoutPolicy(run_timeout=55),
    )
    builder.add_node("validation", validation_node)
    builder.add_node("revision", revision_node)

    builder.add_edge(START, "data_preparation")
    builder.add_edge("data_preparation", "forecasting")
    builder.add_edge("forecasting", "validation")
    builder.add_conditional_edges("validation", should_revise, ["revision", END])
    builder.add_edge("revision", "forecasting")

    return builder.compile(checkpointer=MemorySaver())
