from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ServiceError(Exception):
    """Stable application error transported to the desktop UI."""

    code: str
    message: str
    stage: str = "service"
    target: str | None = None
    details: dict[str, Any] | None = None
    side_effect_possible: bool = False
    next_action: str | None = None

    def __str__(self) -> str:
        return self.message

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "target": self.target,
            "details": self.details or {},
            "side_effect_possible": self.side_effect_possible,
            "next_action": self.next_action,
        }


def invalid(message: str, *, stage: str, target: str | None = None, **details: Any) -> ServiceError:
    return ServiceError("INVALID_REQUEST", message, stage, target, details)

