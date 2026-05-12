from app.ml.preprocessing import build_feature_frame, impute_missing


def test_impute_missing_handles_nulls():
    import pandas as pd

    df = pd.DataFrame(
        {"timestamp": ["2024-01-01"], "consumption_kwh": [None]}
    )
    result = impute_missing(df)
    assert result["consumption_kwh"].isna().sum() == 0


def test_build_feature_frame_requires_columns():
    import pandas as pd

    df = pd.DataFrame({"timestamp": ["2024-01-01"]})
    try:
        build_feature_frame(df)
    except ValueError as exc:
        assert "consumption_kwh" in str(exc)
