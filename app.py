"""Valura AI Arena — FastAPI service entry point.

Three endpoints:
  GET  /health    → 200 once data is loaded and service is ready
  GET  /agents    → agent roster (schema/agents.schema.json)
  POST /answer    → one question envelope in, one answer object out
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from takehome_service.config import settings
from takehome_service.roster import get_roster
from takehome_service.validation import SchemaValidator, SchemaValidationError
from takehome_service.data import DataAccessError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepQuery Service", version="1.0.0")

answer_validator = SchemaValidator("schema/answer.schema.json")
agents_validator = SchemaValidator("schema/agents.schema.json")


@app.on_event("startup")
def startup_event() -> None:
    settings.initialize()
    logger.info("Service ready — data loaded, agents initialised.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "loaded": settings.loaded}


@app.get("/agents")
def agents() -> JSONResponse:
    roster = get_roster()
    try:
        agents_validator.validate(roster)
    except SchemaValidationError as exc:
        return JSONResponse(status_code=500, content={"error": exc.errors})
    return JSONResponse(status_code=200, content=roster)


@app.post("/answer")
async def answer(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "request body must be valid JSON"})

    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": "payload must be a JSON object"})

    missing = [f for f in ("question_id", "client_id", "prompt") if not payload.get(f)]
    if missing:
        return JSONResponse(status_code=400, content={"error": f"missing required fields: {missing}"})

    if settings.answer_service is None:
        return JSONResponse(status_code=503, content={"error": "service not ready"})

    try:
        response = settings.answer_service.answer(payload)
    except DataAccessError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        logger.exception("Unhandled error answering question %s", payload.get("question_id"))
        return JSONResponse(status_code=500, content={"error": str(exc)})

    try:
        answer_validator.validate(response)
    except SchemaValidationError as exc:
        logger.error("Schema validation failed for %s: %s", payload.get("question_id"), exc.errors)
        return JSONResponse(status_code=500, content={"error": exc.errors, "response": response})

    return JSONResponse(status_code=200, content=response)
