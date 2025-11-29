import os
import aiofiles
from uuid import uuid4, UUID
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from geoalchemy2.shape import to_shape

from app.database import get_db
from app.dependencies import get_current_user, RoleChecker
from app.models.user_family_models import User
from app.repositories.incident_repository import IncidentRepository
from app.schemas.incidents import (
    IncidentCreateRequest,
    IncidentResponse,
    IncidentStatusUpdate,
    MediaResponse,
    IncidentCreateRequest as IncidentUpdateRequest,
)

router = APIRouter(prefix="/incidents", tags=["Incidents & SOS"])

UPLOAD_DIR = "app/static/uploads/incidents"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def format_incident_response(incident):
    try:
        shape = to_shape(incident.location)
        lat, lon = shape.y, shape.x
    except:
        lat, lon = 0.0, 0.0

    return IncidentResponse(
        incident_id=incident.incident_id,
        reported_by_user_id=incident.reported_by_user_id,
        title=incident.title,
        description=incident.description,
        incident_type=incident.incident_type,
        status=incident.status,
        reported_at=incident.reported_at,
        latitude=lat,
        longitude=lon,
        media=[
            MediaResponse(
                media_id=m.media_id,
                file_type=m.file_type,
                url=f"/static/uploads/incidents/{os.path.basename(m.storage_path)}",
            )
            for m in incident.media
        ],
    )


# --- Endpoints ---

@router.post("", response_model=IncidentResponse)
async def create_incident(
    payload: IncidentCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Unified Endpoint: SOS (no fields) OR Report (fields).
    """
    try:
        repo = IncidentRepository(db)

        # 1. Check Duplicates
        check_type = payload.incident_type if payload.incident_type else "sos"
        duplicate = await repo.find_duplicate_incident(
            lat=payload.latitude, lon=payload.longitude, incident_type=check_type
        )

        if duplicate:
            return format_incident_response(duplicate)

        # 2. Create
        is_sos = payload.title is None
        new_incident = await repo.create_incident(user_id=current_user.user_id, data=payload, is_sos=is_sos)

        # 3. Trigger SOS Logic
        if is_sos or payload.incident_type == "sos":
            # Phase 4: WebSocket broadcast to Commanders
            pass

        return format_incident_response(new_incident)
    except Exception as e:
        import traceback

        print(f"❌ Error creating incident: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to create incident: {str(e)}")


@router.post("/{incident_id}/media")
async def upload_media(
    incident_id: UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    mime = file.content_type
    if mime.startswith("image/"):
        file_type = "image"
    elif mime.startswith("audio/"):
        file_type = "audio"
    elif mime.startswith("video/"):
        file_type = "video"
    else:
        raise HTTPException(400, "Invalid file type")

    ext = file.filename.split(".")[-1]
    filename = f"{incident_id}_{uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    repo = IncidentRepository(db)
    media_entry = await repo.add_media(
        incident_id=incident_id,
        user_id=current_user.user_id,
        file_meta={
            "file_type": file_type,
            "mime_type": mime,
            "storage_path": file_path,
        },
    )

    return {"media_id": media_entry.media_id, "url": f"/static/uploads/incidents/{filename}"}


@router.get("", response_model=List[IncidentResponse])
async def get_incidents(current_user: User = Depends(RoleChecker(["commander"])), db: AsyncSession = Depends(get_db)):
    repo = IncidentRepository(db)
    incidents = await repo.get_all_open_incidents()
    return [format_incident_response(i) for i in incidents]


@router.get("/mine", response_model=List[IncidentResponse])
async def get_my_incidents(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    repo = IncidentRepository(db)
    incidents = await repo.get_incidents_for_user(current_user.user_id)
    return [format_incident_response(i) for i in incidents]


@router.patch("/{incident_id}/status")
async def update_incident_status(
    incident_id: UUID,
    payload: IncidentStatusUpdate,
    current_user: User = Depends(RoleChecker(["commander"])),
    db: AsyncSession = Depends(get_db),
):
    repo = IncidentRepository(db)

    if payload.status == "discarded":
        updated = await repo.discard_incident(incident_id)
        if not updated:
            raise HTTPException(404, "Incident not found")
        return {"message": "Incident discarded"}

    elif payload.status == "converted":
        # TRIGGER THE DISASTER CREATION LOGIC
        disaster = await repo.convert_to_disaster(
            incident_id, severity=payload.severity_level, disaster_type=payload.disaster_type
        )
        if not disaster:
            raise HTTPException(404, "Incident not found or already converted")

        return {
            "message": "Incident converted to Disaster",
            "disaster_id": disaster.disaster_id if hasattr(disaster, "disaster_id") else None,
        }

    else:
        # Just a generic status update if needed (e.g. 'open')
        return {"message": "No action taken"}


@router.patch("/{incident_id}", response_model=IncidentResponse)
async def update_incident(
    incident_id: UUID,
    payload: IncidentUpdateRequest,
    current_user: User = Depends(RoleChecker(["commander"])),
    db: AsyncSession = Depends(get_db),
):
    repo = IncidentRepository(db)
    updated = await repo.update_incident(incident_id, payload.model_dump(exclude_none=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Incident not found")
    return format_incident_response(updated)


@router.delete("/{incident_id}")
async def delete_incident(
    incident_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = IncidentRepository(db)
    incident = await repo.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    is_commander = current_user.role.name == "commander" if current_user.role else False
    if not is_commander and incident.reported_by_user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to delete this incident")

    await repo.delete_incident(incident_id)
    return {"message": "Incident deleted"}
import os
import aiofiles
from uuid import uuid4, UUID
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from geoalchemy2.shape import to_shape

from app.database import get_db
from app.dependencies import get_current_user, RoleChecker
from app.models.user_family_models import User
from app.repositories.incident_repository import IncidentRepository
from app.schemas.incidents import (
    IncidentCreateRequest, 
    IncidentResponse, 
    IncidentStatusUpdate,
    MediaResponse,
    IncidentCreateRequest as IncidentUpdateRequest
)

router = APIRouter(prefix="/incidents", tags=["Incidents & SOS"])

UPLOAD_DIR = "app/static/uploads/incidents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def format_incident_response(incident):
    try:
        shape = to_shape(incident.location)
        lat, lon = shape.y, shape.x
    except:
        lat, lon = 0.0, 0.0

    return IncidentResponse(
        incident_id=incident.incident_id,
        reported_by_user_id=incident.reported_by_user_id,
        title=incident.title,
        description=incident.description,
        incident_type=incident.incident_type,
        status=incident.status,
        reported_at=incident.reported_at,
        latitude=lat,
        longitude=lon,
        media=[
            MediaResponse(
                media_id=m.media_id, 
                file_type=m.file_type, 
                url=f"/static/uploads/incidents/{os.path.basename(m.storage_path)}"
            ) for m in incident.media
        ]
    )

# --- Endpoints ---

@router.post("", response_model=IncidentResponse)
async def create_incident(
    payload: IncidentCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Unified Endpoint: SOS (no fields) OR Report (fields).
    """
    try:
        repo = IncidentRepository(db)

        # 1. Check Duplicates
        check_type = payload.incident_type if payload.incident_type else "sos"
        duplicate = await repo.find_duplicate_incident(
            lat=payload.latitude,
            lon=payload.longitude,
            incident_type=check_type
        )

        if duplicate:
            return format_incident_response(duplicate)

        # 2. Create
        is_sos = (payload.title is None)
        new_incident = await repo.create_incident(
            user_id=current_user.user_id, 
            data=payload,
            is_sos=is_sos
        )

        # 3. Trigger SOS Logic
        if is_sos or payload.incident_type == 'sos':
            # Phase 4: WebSocket broadcast to Commanders
            pass

        return format_incident_response(new_incident)
    except Exception as e:
        import traceback
        print(f"❌ Error creating incident: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to create incident: {str(e)}")


from app.services.inference_service import run_inference_and_update_db

@router.post("/{incident_id}/media")
async def upload_media(
    incident_id: UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    mime = file.content_type
    if mime.startswith("image/"):
        file_type = "image"
    elif mime.startswith("audio/"):
        file_type = "audio"
    elif mime.startswith("video/"):
        file_type = "video"
    else:
        raise HTTPException(400, "Invalid file type")

    ext = file.filename.split(".")[-1]
    filename = f"{incident_id}_{uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    repo = IncidentRepository(db)
    media_entry = await repo.add_media(
        incident_id=incident_id,
        user_id=current_user.user_id,
        file_meta={
            "file_type": file_type,
            "mime_type": mime,
            "storage_path": file_path
        }
    )

    if file_type == "image":
        background_tasks.add_task(run_inference_and_update_db, media_entry.media_id)

    return {
        "media_id": media_entry.media_id,
        "url": f"/static/uploads/incidents/{filename}"
    }


@router.get("", response_model=List[IncidentResponse])
async def get_incidents(
    current_user: User = Depends(RoleChecker(["commander"])),
    db: AsyncSession = Depends(get_db)
):
    repo = IncidentRepository(db)
    incidents = await repo.get_all_open_incidents()
    return [format_incident_response(i) for i in incidents]


@router.get("/mine", response_model=List[IncidentResponse])
async def get_my_incidents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    repo = IncidentRepository(db)
    incidents = await repo.get_incidents_for_user(current_user.user_id)
    return [format_incident_response(i) for i in incidents]


@router.patch("/{incident_id}/status")
async def update_incident_status(
    incident_id: UUID,
    payload: IncidentStatusUpdate,
    current_user: User = Depends(RoleChecker(["commander"])),
    db: AsyncSession = Depends(get_db)
):
    repo = IncidentRepository(db)

    if payload.status == 'discarded':
        updated = await repo.discard_incident(incident_id)
        if not updated:
            raise HTTPException(404, "Incident not found")
        return {"message": "Incident discarded"}

    elif payload.status == 'converted':
        # TRIGGER THE DISASTER CREATION LOGIC
        disaster = await repo.convert_to_disaster(
            incident_id, 
            severity=payload.severity_level, 
            disaster_type=payload.disaster_type
        )
        if not disaster:
            raise HTTPException(404, "Incident not found or already converted")
        
        return {
            "message": "Incident converted to Disaster",
            "disaster_id": disaster.disaster_id if hasattr(disaster, 'disaster_id') else None
        }
    
    else:
        # Just a generic status update if needed (e.g. 'open')
        return {"message": "No action taken"}


@router.patch("/{incident_id}", response_model=IncidentResponse)
async def update_incident(
    incident_id: UUID,
    payload: IncidentUpdateRequest,
    current_user: User = Depends(RoleChecker(["commander"])),
    db: AsyncSession = Depends(get_db)
):
    repo = IncidentRepository(db)
    updated = await repo.update_incident(
        incident_id,
        payload.model_dump(exclude_none=True),
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Incident not found")
    return format_incident_response(updated)


@router.delete("/{incident_id}")
async def delete_incident(
    incident_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    repo = IncidentRepository(db)
    incident = await repo.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    is_commander = current_user.role.name == "commander" if current_user.role else False
    if not is_commander and incident.reported_by_user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to delete this incident")

    await repo.delete_incident(incident_id)
    return {"message": "Incident deleted"}
