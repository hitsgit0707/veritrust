"""Health check schema for VeriTrust AI."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Health check status response payload."""
    status: str
    app_name: str
    environment: str
    database: str
    llm_provider: str
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)
