import json


def build_prompt(context: dict, params: dict, revision_note: str = "") -> str:
    return f"""
Historical consumption summary:
{json.dumps(context, indent=2)}

Forecast request:
- Start date: {params['start_date']}
- Horizon: {params['horizon_days']} days
- Resolution: {params['resolution']}

Generate hourly/daily/weekly predictions from the start date covering the full horizon.
Each prediction must have a timestamp (ISO 8601) and value (float, kWh).
{revision_note}
"""
