from typing import TypedDict


class AgentState(TypedDict):
    dataset_path: str
    forecast_params: dict
    agent_run_id: str
    context_json: dict
    llm_response: dict | None
    predictions: list[dict] | None
    validation_report: dict | None
    revision_count: int
    reasoning: str | None
    status: str
    error: str | None
