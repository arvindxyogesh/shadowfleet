from datetime import datetime, timezone

from sqlalchemy.orm import Session

from .db import AuditLogEntry


def log_audit(session: Session, actor: str, action: str, details: dict | None = None) -> AuditLogEntry:
    entry = AuditLogEntry(
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        actor=actor,
        action=action,
        details=details or {},
    )
    session.add(entry)
    return entry
