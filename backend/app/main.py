from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db
from app.api import tasks, models, configs, data_sources, uploads, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables
    await init_db()
    yield
    # Shutdown


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8100",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(tasks.router, prefix=settings.API_PREFIX)
app.include_router(models.router, prefix=settings.API_PREFIX)
app.include_router(configs.router, prefix=settings.API_PREFIX)
app.include_router(data_sources.router, prefix=settings.API_PREFIX)
app.include_router(uploads.router, prefix=settings.API_PREFIX)
app.include_router(predict.router, prefix=settings.API_PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.APP_VERSION}
