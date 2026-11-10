"""FastAPI interface for local claim-image research analyses."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from claimguard import __version__
from claimguard.config import get_settings
from claimguard.schemas import AnalysisReport, ExperimentRecord
from claimguard.service import analyze_and_store
from claimguard.storage import ResearchStore

LOGGER = logging.getLogger(__name__)
settings = get_settings()
store = ResearchStore(settings.database_path)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO)
    yield


app = FastAPI(
    title="ClaimGuard Research API",
    description="Local research prototype for image provenance and manipulation cues.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "version": __version__,
        "ai_detector_configured": bool(settings.model_path),
        "fusion_model_configured": settings.fusion_model_path.is_file(),
    }


@app.post("/api/analyze", response_model=AnalysisReport)
async def analyze(file: Annotated[UploadFile, File()]) -> AnalysisReport:
    media_type = (file.content_type or "").lower()
    if media_type not in settings.allowed_mime_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image media type: {media_type or 'unknown'}.",
        )
    filename = _safe_filename(file.filename or "uploaded-image")
    try:
        image_bytes = await _read_limited(file, settings.max_upload_bytes)
        return await run_in_threadpool(
            analyze_and_store,
            image_bytes,
            filename,
            media_type,
            settings,
            store,
        )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Image analysis failed for %s", filename)
        raise HTTPException(
            status_code=500, detail="Image analysis failed; see server logs."
        ) from error
    finally:
        await file.close()


@app.get("/api/analyses", response_model=list[AnalysisReport])
def analyses(limit: int = Query(default=50, ge=1, le=500)) -> list[AnalysisReport]:
    return store.list_analyses(limit)


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisReport)
def get_analysis(analysis_id: str) -> AnalysisReport:
    report = store.get_analysis(analysis_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return report


@app.get("/api/experiments", response_model=list[ExperimentRecord])
def experiments(limit: int = Query(default=50, ge=1, le=500)) -> list[ExperimentRecord]:
    return store.list_experiments(limit)


@app.exception_handler(Exception)
async def unexpected_error_handler(_, error: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled API error", exc_info=error)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(min(1024 * 1024, limit + 1 - total)):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413, detail=f"Image exceeds the {limit}-byte upload limit."
            )
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    return b"".join(chunks)


def _safe_filename(filename: str) -> str:
    cleaned = filename.replace("\\", "/").split("/")[-1].strip()
    return cleaned[:255] or "uploaded-image"
