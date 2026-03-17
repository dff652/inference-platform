import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.core.config import settings

router = APIRouter(prefix="/uploads", tags=["File Uploads"])

UPLOAD_DIR = Path(settings.DATA_ROOT) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".txt", ".xlsx", ".xls"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.post("")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload CSV files for inference tasks. Returns server-side file paths."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Create a unique batch directory
    batch_id = uuid.uuid4().hex[:12]
    batch_dir = UPLOAD_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        # Validate extension
        suffix = Path(f.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{suffix}' not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        # Validate size
        content = await f.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}' exceeds {MAX_FILE_SIZE // 1024 // 1024}MB limit",
            )

        # Save with original filename (sanitized)
        safe_name = Path(f.filename).name  # strip directory components
        dest = batch_dir / safe_name

        # Handle name collision
        if dest.exists():
            stem = dest.stem
            dest = batch_dir / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"

        dest.write_bytes(content)
        saved.append({
            "filename": f.filename,
            "path": str(dest),
            "size": len(content),
        })

    return JSONResponse(content={
        "batch_id": batch_id,
        "files": saved,
    })
