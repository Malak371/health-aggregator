from dataclasses import dataclass, field

from app.schemas import DailyRecord


@dataclass
class ParseResult:
    records: list[DailyRecord]
    warnings: list[str] = field(default_factory=list)
