"""FastAPI dashboard: live MJPEG view plus streaming metrics."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from visionbench import __version__
from visionbench import report as report_mod
from visionbench.draw import placeholder
from visionbench.hostinfo import default_profile, host_info
from visionbench.metrics.system import detect_gpu_backend
from visionbench.models import SPECS
from visionbench.models.base import resolve_device
from visionbench.runner import RunConfig
from visionbench.web.session import SessionStore

STATIC = Path(__file__).parent / "static"
BOUNDARY = "visionbenchframe"

app = FastAPI(title="ai-vision-detector-bench", version=__version__)
store = SessionStore()

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


class StartRequest(BaseModel):
    source: str = Field(..., description="File path, RTSP URL, or MJPEG URL.")
    model: str = "dfine-nano"
    duration_s: float = 60.0
    device: str = "auto"
    confidence: float = 0.3
    classes: list[str] | None = None
    mode: str = "realtime"
    imgsz: int | None = None
    precision: str = "fp32"
    profile: str | None = None
    pace_fps: float | None = None


@app.get("/")
async def index() -> FileResponse:
    page = STATIC / "index.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="static/index.html is missing")
    return FileResponse(page)


@app.get("/api/host")
async def api_host() -> dict[str, Any]:
    device = resolve_device("auto")
    backend, _ = detect_gpu_backend(device)
    return {
        "profile": default_profile(),
        "device": device,
        "host": host_info(),
        "gpu_telemetry": {
            "kind": backend.kind,
            "name": backend.name,
            "utilisation_available": backend.utilisation_available,
            "power_available": backend.power_available,
            "note": backend.note,
        },
    }


@app.get("/api/models")
async def api_models() -> list[dict[str, Any]]:
    return [
        {
            "key": spec.key,
            "name": spec.display_name,
            "family": spec.family,
            "license": spec.license,
            "license_url": spec.license_url,
            "commercial": spec.commercial,
            "permissive": spec.permissive,
            "weights": spec.weights,
            "extra": spec.extra,
            "notes": spec.notes,
        }
        for spec in SPECS
    ]


@app.get("/api/results")
async def api_results(limit: int = 50) -> dict[str, Any]:
    results = report_mod.load_all()[:limit]
    return {
        "count": len(results),
        "markdown": report_mod.markdown_table(results),
        "rows": [report_mod.to_row(r).__dict__ for r in results],
    }


@app.post("/api/sessions")
async def api_start(request: StartRequest) -> dict[str, Any]:
    config = RunConfig(
        source=request.source,
        model=request.model,
        device=request.device,
        duration_s=request.duration_s,
        confidence=request.confidence,
        classes=request.classes or None,
        mode=request.mode,
        imgsz=request.imgsz,
        precision=request.precision,
        profile=request.profile,
        pace_fps=request.pace_fps,
    )
    session = store.create(config)
    return session.status()


@app.get("/api/sessions")
async def api_sessions() -> list[dict[str, Any]]:
    return [s.status() for s in store.all()]


@app.get("/api/sessions/{session_id}")
async def api_session(session_id: str) -> dict[str, Any]:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="no such session")
    return session.status()


@app.post("/api/sessions/{session_id}/stop")
async def api_stop(session_id: str) -> dict[str, Any]:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="no such session")
    session.stop()
    return {"id": session.id, "state": session.state}


@app.post("/api/sessions/{session_id}/save")
async def api_save(session_id: str) -> dict[str, Any]:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="no such session")
    if session.result is None:
        raise HTTPException(status_code=409, detail="run has not finished")
    path = report_mod.save(session.result)
    return {"saved": str(path)}


@app.get("/api/sessions/{session_id}/stream.mjpg")
async def api_stream(session_id: str) -> StreamingResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="no such session")

    async def frames():
        sent = -1
        idle_deadline = 30.0
        waited = 0.0
        while True:
            jpeg = session.latest_jpeg
            index = session.frame_index
            if jpeg is not None and index != sent:
                sent = index
                waited = 0.0
                yield (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode() + jpeg + b"\r\n"
            else:
                waited += 0.03
            if not session.running:
                # Let the last frame land, then close the multipart stream.
                final = session.latest_jpeg or placeholder(640, 360, session.state)
                if final is not None and session.frame_index != sent:
                    yield (
                        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(final)}\r\n\r\n"
                    ).encode() + final + b"\r\n"
                break
            if waited > idle_deadline:
                break
            await asyncio.sleep(0.03)

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/api/sessions/{session_id}/events")
async def api_events(session_id: str) -> StreamingResponse:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="no such session")

    async def events():
        while True:
            payload = json.dumps(session.status())
            yield f"data: {payload}\n\n"
            if not session.running:
                yield "event: done\ndata: {}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.on_event("shutdown")
async def _shutdown() -> None:
    store.stop_all()
