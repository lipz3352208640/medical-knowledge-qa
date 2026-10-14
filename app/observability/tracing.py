"""Failure-isolated trace context with optional Langfuse HTTP export."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import base64
import json
from app.config import setting
from threading import Lock
import time
import urllib.request
import uuid

from app.observability.metrics import METRICS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TraceContext:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    request_id: str | None = None
    fields: dict[str, object] = field(default_factory=dict)
    spans: list[dict] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock)

    @contextmanager
    def span(self, name: str, **fields: object):
        started = time.perf_counter()
        start_at = _now()
        error: str | None = None
        status = "ok"
        try:
            yield self
        except Exception as exc:
            status = "error"
            error = type(exc).__name__
            raise
        finally:
            value = {
                "trace_id": self.trace_id, "span": name, "started_at": start_at,
                "finished_at": _now(), "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "status": status, "error_code": error, **fields,
            }
            with self._lock:
                self.spans.append(value)

    def finish(self, **fields: object) -> None:
        payload = {"trace_id": self.trace_id, "request_id": self.request_id,
                   "fields": {**self.fields, **fields}, "spans": list(self.spans)}
        _export_async(payload)


def new_trace(*, request_id: str | None = None, **fields: object) -> TraceContext:
    return TraceContext(request_id=request_id, fields=fields)


def _export_async(payload: dict) -> None:
    host = setting("LANGFUSE_HOST")
    public = setting("LANGFUSE_PUBLIC_KEY")
    secret = setting("LANGFUSE_SECRET_KEY")
    if not host or not public or not secret:
        METRICS.inc("telemetry_traces_skipped_total", destination="langfuse")
        return
    # Fire-and-forget through a daemon thread without introducing a queue/MQ.
    from threading import Thread
    Thread(target=_export, args=(host, public, secret, payload), daemon=True).start()


def _export(host: str, public: str, secret: str, payload: dict) -> None:
    try:
        token = base64.b64encode(f"{public}:{secret}".encode()).decode()
        body = json.dumps({"batch": [{"type": "trace-create", "body": payload}]}, ensure_ascii=False).encode()
        request = urllib.request.Request(
            host.rstrip("/") + "/api/public/ingestion", data=body,
            headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2):
            pass
        METRICS.inc("telemetry_exports_total", destination="langfuse")
    except Exception:
        METRICS.inc("telemetry_export_failures_total", destination="langfuse")
