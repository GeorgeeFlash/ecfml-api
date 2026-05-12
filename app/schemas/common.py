from enum import Enum


class ValidationStatus(str, Enum):
    PENDING = "PENDING"
    VALID = "VALID"
    INVALID = "INVALID"
    WARNING = "WARNING"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    WARNING = "WARNING"


class ModelType(str, Enum):
    RANDOM_FOREST = "RANDOM_FOREST"
    SVR = "SVR"


class EngineType(str, Enum):
    RF = "RF"
    SVR = "SVR"
    AGENT = "AGENT"


class Resolution(str, Enum):
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
