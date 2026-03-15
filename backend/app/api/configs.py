from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.config_template import InferenceConfigTemplate
from app.schemas.config_template import ConfigCreate, ConfigUpdate, ConfigResponse, ConfigListResponse

router = APIRouter(prefix="/inference/configs", tags=["Inference Configs"])


SUPPORTED_ALGORITHMS = [
    {"id": 1, "name": "chatts", "display_name": "ChatTS", "description": "ChatTS-8B time series anomaly detection"},
    {"id": 2, "name": "qwen", "display_name": "Qwen-VL", "description": "Qwen-3-VL multimodal detection"},
    {"id": 3, "name": "adtk_hbos", "display_name": "ADTK-HBOS", "description": "Statistical anomaly detection"},
    {"id": 4, "name": "ensemble", "display_name": "Ensemble", "description": "Multi-method voting ensemble"},
    {"id": 5, "name": "wavelet", "display_name": "Wavelet", "description": "Wavelet decomposition analysis"},
    {"id": 6, "name": "isolation_forest", "display_name": "Isolation Forest", "description": "Tree-based anomaly detection"},
]


@router.get("/algorithms")
async def list_algorithms():
    return SUPPORTED_ALGORITHMS


@router.post("", response_model=ConfigResponse, status_code=201)
async def create_config(data: ConfigCreate, db: AsyncSession = Depends(get_db)):
    config = InferenceConfigTemplate(**data.model_dump())
    db.add(config)
    await db.flush()
    await db.refresh(config)
    return config


@router.get("", response_model=ConfigListResponse)
async def list_configs(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    count = (await db.execute(select(sa_func.count(InferenceConfigTemplate.id)))).scalar() or 0
    result = await db.execute(
        select(InferenceConfigTemplate)
        .order_by(InferenceConfigTemplate.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return ConfigListResponse(total=count, items=list(result.scalars().all()))


@router.put("/{config_id}", response_model=ConfigResponse)
async def update_config(config_id: int, data: ConfigUpdate, db: AsyncSession = Depends(get_db)):
    config = await db.get(InferenceConfigTemplate, config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(config, key, value)
    await db.flush()
    await db.refresh(config)
    return config
