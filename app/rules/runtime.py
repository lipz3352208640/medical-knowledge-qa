"""Pin one published rule snapshot for the duration of a request."""

from __future__ import annotations

from contextvars import ContextVar


REQUEST_RULE_SNAPSHOT: ContextVar[dict | None] = ContextVar("request_rule_snapshot", default=None)
