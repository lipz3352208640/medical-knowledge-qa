"""Reliable RAG-domain answer audit persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import uuid

from app.storage.mysql import _connect, _json, parse_mysql_url


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AuditPersistenceError(RuntimeError):
    pass


class MySQLAuditStore:
    def __init__(self, mysql_url: str) -> None:
        self.mysql_url = mysql_url

    def record_answer(
        self, *, request_id: str, trace_id: str | None, question: str,
        answer: str, status: str, evidence: list[dict], rule_version: str | None = None,
        index_version: str | None = None, prompt_version: str | None = None,
        model_id: str | None = None, no_evidence_reason: str | None = None,
        citation_check_status: str | None = None, failure_code: str | None = None,
    ) -> str:
        answer_id = uuid.uuid4().hex
        completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            config = parse_mysql_url(self.mysql_url)
            connection = _connect(config)
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """INSERT INTO qa_audit
                           (answer_id,trace_id,request_id,status,question_digest,answer_digest,
                            cited_evidence,rule_version,index_version,prompt_version,model_id,
                            no_evidence_reason,citation_check_status,completed_at,failure_code)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (answer_id, trace_id, request_id, status, _digest(question), _digest(answer),
                         _json(evidence), rule_version, index_version, prompt_version, model_id,
                         no_evidence_reason, citation_check_status, completed_at, failure_code),
                    )
                connection.commit()
            finally:
                connection.close()
        except Exception as exc:
            raise AuditPersistenceError(str(exc)) from exc
        return answer_id

    def list_answers(self, limit: int = 50) -> list[dict]:
        config = parse_mysql_url(self.mysql_url)
        connection = _connect(config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT answer_id,trace_id,request_id,occurred_at,status,
                              question_digest,answer_digest,cited_evidence,rule_version,
                              index_version,prompt_version,model_id,citation_check_status,
                              completed_at,failure_code
                       FROM qa_audit ORDER BY occurred_at DESC LIMIT %s""",
                    (limit,),
                )
                rows = cursor.fetchall()
            return [self._row(row) for row in rows]
        finally:
            connection.close()

    def get_answer(self, answer_id: str) -> dict | None:
        config = parse_mysql_url(self.mysql_url)
        connection = _connect(config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT answer_id,trace_id,request_id,occurred_at,status,
                              question_digest,answer_digest,cited_evidence,rule_version,
                              index_version,prompt_version,model_id,citation_check_status,
                              completed_at,failure_code
                       FROM qa_audit WHERE answer_id=%s""",
                    (answer_id,),
                )
                row = cursor.fetchone()
            return self._row(row) if row else None
        finally:
            connection.close()

    @staticmethod
    def _row(row: tuple) -> dict:
        names = [
            "answer_id", "trace_id", "request_id", "occurred_at", "status",
            "question_digest", "answer_digest", "cited_evidence", "rule_version",
            "index_version", "prompt_version", "model_id", "citation_check_status",
            "completed_at", "failure_code",
        ]
        value = dict(zip(names, row, strict=False))
        raw = value.get("cited_evidence")
        if isinstance(raw, str):
            value["cited_evidence"] = json.loads(raw)
        return value
