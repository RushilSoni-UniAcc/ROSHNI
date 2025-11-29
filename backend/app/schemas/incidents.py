from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# Request used for creating a new incident (or SOS)
class IncidentCreateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    incident_type: Optional[str] = None
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class MediaResponse(BaseModel):
    media_id: UUID
    file_type: str
    url: str


class IncidentResponse(BaseModel):
    incident_id: UUID
    reported_by_user_id: UUID
    title: Optional[str]
    description: Optional[str]
    incident_type: Optional[str]
    status: str
    reported_at: datetime
    latitude: float
    longitude: float
    media: List[MediaResponse] = []

    class Config:
        from_attributes = True


class IncidentStatusUpdate(BaseModel):
    status: str
    severity_level: Optional[str] = None
    disaster_type: Optional[str] = None


# For updates we reuse the same fields as creation (all optional on update)
class IncidentUpdateRequest(IncidentCreateRequest):
    title: Optional[str] = None
    description: Optional[str] = None
