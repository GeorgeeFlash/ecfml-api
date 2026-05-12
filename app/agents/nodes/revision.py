from langgraph.config import get_stream_writer

from app.agents.state import AgentState


async def revision_node(state: AgentState) -> dict:
    writer = get_stream_writer()
    writer(
        {
            "type": "node_start",
            "node": "revision",
            "message": "Revising forecast...",
        }
    )

    revision_count = state.get("revision_count", 0) + 1
    writer({"type": "node_complete", "node": "revision"})
    return {"revision_count": revision_count, "status": "revising"}
