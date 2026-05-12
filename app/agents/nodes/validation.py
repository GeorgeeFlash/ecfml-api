from langgraph.config import get_stream_writer

from app.agents.state import AgentState


async def validation_node(state: AgentState) -> dict:
    writer = get_stream_writer()
    writer(
        {
            "type": "node_start",
            "node": "validation",
            "message": "Validating forecast...",
        }
    )

    predictions = state.get("predictions") or []
    ctx = state.get("context_json", {})
    mean = ctx.get("mean_kwh", 0)
    std = ctx.get("std_kwh", 0)

    anomalies = []
    for idx, item in enumerate(predictions):
        value = float(item.get("value", 0))
        if value < 0 or abs(value - mean) > 3 * std:
            anomalies.append(idx)

    anomaly_pct = (len(anomalies) / max(len(predictions), 1)) * 100
    report = {
        "passed": anomaly_pct <= 10,
        "anomaly_pct": round(anomaly_pct, 2),
        "failed_indices": anomalies,
    }

    writer({"type": "node_complete", "node": "validation"})

    return {"validation_report": report, "status": "validated"}
