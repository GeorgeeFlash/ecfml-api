from __future__ import annotations

from app.schemas.common import ValidationStatus
from app.schemas.dataset import DatasetPreviewResponse, DatasetValidationReport
from app.services.data_utils import load_dataset

DEFAULT_REQUIRED_COLUMNS = ["timestamp", "consumption_kwh"]


def validate_dataset(
    file_url: str, required_columns: list[str] | None = None
) -> DatasetValidationReport:
    required = required_columns or DEFAULT_REQUIRED_COLUMNS
    df = load_dataset(file_url)
    columns = list(df.columns)

    missing = [col for col in required if col not in columns]
    status = ValidationStatus.VALID if not missing else ValidationStatus.INVALID
    warnings = []
    if status == ValidationStatus.VALID and len(df) < 100:
        status = ValidationStatus.WARNING
        warnings.append("Dataset has fewer than 100 rows.")

    return DatasetValidationReport(
        status=status,
        missing_columns=missing,
        columns=columns,
        row_count=len(df),
        warnings=warnings,
        details=None,
    )


def preview_dataset(file_url: str, rows: int = 100) -> DatasetPreviewResponse:
    df = load_dataset(file_url)
    subset = df.head(rows)
    return DatasetPreviewResponse(
        columns=list(subset.columns),
        rows=subset.to_dict(orient="records"),
        row_count=len(df),
    )
