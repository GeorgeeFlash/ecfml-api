from app.agents.state import AgentState


def test_agent_state_shape():
    state: AgentState = {
        "dataset_path": "path",
        "forecast_params": {},
        "agent_run_id": "id",
        "context_json": {},
        "llm_response": None,
        "predictions": None,
        "validation_report": None,
        "revision_count": 0,
        "reasoning": None,
        "status": "started",
        "error": None,
    }
    assert state["dataset_path"] == "path"
