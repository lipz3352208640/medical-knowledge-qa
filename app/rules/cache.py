"""Local, atomic cache for the management service's published rule snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


class RuleSnapshotCache:
    def __init__(self, path: str = "data/rule_snapshot.json", *, admin_url: str | None = None,
                 token: str | None = None) -> None:
        self.path = Path(path)
        self.admin_url = admin_url
        self.token = token
        self.snapshot: dict | None = None
        self.version: str | None = None
        self.content_hash: str | None = None
        self.error: str | None = None
        self.load()

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            snapshot = payload["snapshot"]
            content_hash = payload["content_hash"]
            if self._hash(snapshot) != content_hash:
                raise ValueError("规则快照哈希校验失败")
            self.snapshot = snapshot
            self.version = payload.get("version")
            self.content_hash = content_hash
            self.error = None
            return snapshot
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.error = str(exc)
            return None

    def refresh(self) -> dict | None:
        if not self.admin_url:
            return self.snapshot
        url = self.admin_url.rstrip("/") + "/internal/v1/rule-publications/active"
        headers = {"Accept": "application/json"}
        if self.token:
            headers["X-Service-Token"] = self.token
        try:
            with urlopen(Request(url, headers=headers), timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
            snapshot = payload["snapshot"]
            content_hash = payload["content_hash"]
            if self._hash(snapshot) != content_hash:
                raise ValueError("远程规则快照哈希校验失败")
            if content_hash != self.content_hash:
                self._write(payload.get("version"), content_hash, snapshot)
            self.snapshot = snapshot
            self.version = payload.get("version")
            self.content_hash = content_hash
            self.error = None
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.error = str(exc)
        return self.snapshot

    def _write(self, version: str | None, content_hash: str, snapshot: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": version, "content_hash": content_hash, "snapshot": snapshot,
                   "saved_at": datetime.now(timezone.utc).isoformat()}
        with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                               prefix=self.path.name + ".", delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            temporary = handle.name
        os.replace(temporary, self.path)

    @staticmethod
    def _hash(snapshot: dict) -> str:
        return hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
