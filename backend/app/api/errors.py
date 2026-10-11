"""One error shape for every failure: {"detail": {"message": ..., "errors": [...]}}."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.ingestion.errors import IngestionError


def _body(message: str, errors: list[str] | None = None) -> dict[str, object]:
    return {"detail": {"message": message, "errors": errors or []}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(IngestionError)
    async def ingestion_error(_: Request, exc: IngestionError) -> JSONResponse:
        return JSONResponse(_body(exc.message, exc.errors), status_code=422)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
        return JSONResponse(_body("The request is not valid.", errors), status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            _body(str(exc.detail)), status_code=exc.status_code, headers=exc.headers
        )
