import json

from pydantic import BaseModel
from langgraph.config import get_stream_writer

from app.agents.llm import get_llm
from app.agents.state import AgentState


class ForecastOutput(BaseModel):
    predictions: list[dict]
    reasoning: str
    confidence: str


SYSTEM_PROMPT = (
    "You are an expert electricity demand forecasting assistant. "
    "You will be given historical consumption statistics and asked to generate a "
    "structured short-term or medium-term forecast for the North West Region of Cameroon. "
    "Return ONLY valid structured data matching the required schema. "
    "Base your forecast on the provided statistics, seasonality patterns, and context. "
    "Do not invent data not present in the context."
)


async def forecasting_node(state: AgentState) -> dict:
    writer = get_stream_writer()
    writer(
        {
            "type": "node_start",
            "node": "forecasting",
            "message": "Generating forecast with LLM...",
        }
    )

    ctx = state["context_json"]
    params = state["forecast_params"]
    revision_count = state.get("revision_count", 0)

    revision_note = ""
    if revision_count > 0:
        report = state.get("validation_report", {})
        revision_note = (
            f"\nIMPORTANT: Your previous forecast had {report.get('anomaly_pct', 0):.1f}% "
            f"anomalous predictions (outside ±3 std of historical mean = {ctx['mean_kwh']} ± "
            f"{ctx['std_kwh'] * 3:.2f} kWh). Please correct this in your revised forecast."
        )

    user_prompt = f"""
Historical consumption summary:
{json.dumps(ctx, indent=2)}

Forecast request:
- Start date: {params['start_date']}
- Horizon: {params['horizon_days']} days
- Resolution: {params['resolution']}

Generate hourly/daily/weekly predictions from the start date covering the full horizon.
Each prediction must have a timestamp (ISO 8601) and value (float, kWh).
{revision_note}
"""

    llm = get_llm()
    structured_llm = llm.with_structured_output(ForecastOutput)
    response = await structured_llm.ainvoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    writer({"type": "node_complete", "node": "forecasting"})

    return {
        "llm_response": response.model_dump(),
        "predictions": response.predictions,
        "reasoning": response.reasoning,
        "status": "forecasted",
    }
