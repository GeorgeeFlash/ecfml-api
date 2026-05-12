import pandas as pd
from langgraph.config import get_stream_writer

from app.agents.state import AgentState


async def data_preparation_node(state: AgentState) -> dict:
    writer = get_stream_writer()
    writer(
        {
            "type": "node_start",
            "node": "data_preparation",
            "message": "Loading and summarizing consumption data...",
        }
    )

    df = pd.read_parquet(state["dataset_path"])
    if "consumption_kwh" not in df.columns:
        raise ValueError("Processed dataset must include consumption_kwh")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    consumption = df["consumption_kwh"].astype(float)
    context = {
        "row_count": int(len(df)),
        "date_range": [str(df.index.min()), str(df.index.max())],
        "mean_kwh": round(float(consumption.mean()), 2),
        "std_kwh": round(float(consumption.std()), 2),
        "min_kwh": round(float(consumption.min()), 2),
        "max_kwh": round(float(consumption.max()), 2),
        "recent_24h": consumption.tail(24).tolist(),
        "recent_7d_daily_means": consumption.tail(168).resample("D").mean().tolist(),
        "rolling_mean_24h": round(float(consumption.tail(24).mean()), 2),
        "rolling_mean_7d": round(float(consumption.tail(168).mean()), 2),
    }

    writer({"type": "node_complete", "node": "data_preparation"})
    return {"context_json": context, "status": "data_prepared"}
