"""Errors raised by the ingestion parsers."""


class IngestionError(ValueError):
    """The uploaded file cannot be used. `errors` lists the specific problems."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.errors = errors or []
