from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.data_source import DataSource, DataSourceStatus
from app.schemas.data_source import (
    DataSourceCreate,
    DataSourceUpdate,
    DataSourceResponse,
    DataSourceListResponse,
    DataSourceTestResult,
)

router = APIRouter(prefix="/data-sources", tags=["Data Sources"])


@router.post("", response_model=DataSourceResponse, status_code=201)
async def create_data_source(data: DataSourceCreate, db: AsyncSession = Depends(get_db)):
    ds = DataSource(**data.model_dump())
    db.add(ds)
    await db.flush()
    await db.refresh(ds)
    return ds


@router.get("", response_model=DataSourceListResponse)
async def list_data_sources(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    count = (await db.execute(select(sa_func.count(DataSource.id)))).scalar() or 0
    result = await db.execute(
        select(DataSource).order_by(DataSource.updated_at.desc()).offset(offset).limit(limit)
    )
    return DataSourceListResponse(total=count, items=list(result.scalars().all()))


@router.get("/{ds_id}", response_model=DataSourceResponse)
async def get_data_source(ds_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(DataSource, ds_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Data source not found")
    return ds


@router.put("/{ds_id}", response_model=DataSourceResponse)
async def update_data_source(ds_id: int, data: DataSourceUpdate, db: AsyncSession = Depends(get_db)):
    ds = await db.get(DataSource, ds_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Data source not found")
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(ds, key, value)
    await db.flush()
    await db.refresh(ds)
    return ds


@router.post("/{ds_id}/test", response_model=DataSourceTestResult)
async def test_data_source(ds_id: int, db: AsyncSession = Depends(get_db)):
    ds = await db.get(DataSource, ds_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Data source not found")

    # TODO: implement actual connectivity tests per data source type
    import time
    start = time.monotonic()
    try:
        # Placeholder: test connectivity based on type
        success = True
        message = "Connection successful"
    except Exception as e:
        success = False
        message = str(e)
    latency = (time.monotonic() - start) * 1000

    ds.status = DataSourceStatus.ACTIVE if success else DataSourceStatus.ERROR
    ds.last_check_at = datetime.utcnow()
    await db.flush()

    return DataSourceTestResult(success=success, message=message, latency_ms=latency)
