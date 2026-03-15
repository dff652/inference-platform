from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.model_entity import ModelStatus
from app.schemas.model_entity import ModelCreate, ModelUpdate, ModelResponse, ModelListResponse
from app.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["Model Center"])


@router.post("", response_model=ModelResponse, status_code=201)
async def create_model(data: ModelCreate, db: AsyncSession = Depends(get_db)):
    return await ModelService.create(db, data)


@router.get("", response_model=ModelListResponse)
async def list_models(
    family: str | None = None,
    status: ModelStatus | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    items, total = await ModelService.list_models(db, family, status, offset, limit)
    return ModelListResponse(total=total, items=items)


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int, db: AsyncSession = Depends(get_db)):
    model = await ModelService.get(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(model_id: int, data: ModelUpdate, db: AsyncSession = Depends(get_db)):
    model = await ModelService.update(db, model_id, data)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/{model_id}/activate", response_model=ModelResponse)
async def activate_model(model_id: int, db: AsyncSession = Depends(get_db)):
    model = await ModelService.set_status(db, model_id, ModelStatus.ACTIVE)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/{model_id}/archive", response_model=ModelResponse)
async def archive_model(model_id: int, db: AsyncSession = Depends(get_db)):
    model = await ModelService.set_status(db, model_id, ModelStatus.ARCHIVED)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model
