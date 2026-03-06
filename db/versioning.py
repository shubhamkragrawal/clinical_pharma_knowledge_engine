"""
db/versioning.py

Versioning layer for the FDA Clinical Pharmacology Pipeline.

Rules enforced here:
- A new content_versions row is ALWAYS written before any edit to
  qna_store or comments_store.
- The existing record's is_current flag is set to FALSE before the
  updated record is written.
- version_number increments per (record_id, record_type) pair.
- This module never commits directly — it expects to be called
  inside a session context managed by the caller or two_phase_commit.py.

Valid record_type values:  'qna', 'comment'
Valid changed_by values:   'curator', 'llm', 'system'
"""

from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy.orm import Session

from db.models import CommentsStore, ContentVersion, QnAStore
from utils.logger import get_module_logger

logger = get_module_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_next_version_number(
    session: Session,
    record_id: int,
    record_type: str,
) -> int:
    """
    Return the next sequential version number for the given record.
    Queries content_versions to find the current maximum and adds 1.
    Returns 1 if no prior versions exist.
    """
    existing_max_version = (
        session.query(ContentVersion.version_number)
        .filter(
            ContentVersion.record_id == record_id,
            ContentVersion.record_type == record_type,
        )
        .order_by(ContentVersion.version_number.desc())
        .first()
    )
    if existing_max_version is None:
        return 1
    return existing_max_version[0] + 1


def _build_qna_snapshot(qna_record: QnAStore) -> Dict[str, Any]:
    """
    Build the JSONB content_snapshot for a QnA record.
    Shape: {"question": "...", "answer": "..."}
    """
    return {
        "question": qna_record.question,
        "answer": qna_record.answer,
    }


def _build_comment_snapshot(comment_record: CommentsStore) -> Dict[str, Any]:
    """
    Build the JSONB content_snapshot for a comment record.
    Shape: {"header": "...", "comment": "..."}
    """
    return {
        "header": comment_record.header,
        "comment": comment_record.comment,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_initial_version_for_qna(
    session: Session,
    qna_record: QnAStore,
    changed_by: str,
    change_reason: str,
) -> ContentVersion:
    """
    Write version 1 for a newly created QnA record.
    Called during ingestion immediately after the qna_store row is flushed
    (so qna_id is available) but before the session commits.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session. Caller controls commit/rollback.
    qna_record : QnAStore
        The freshly created QnA row with a valid qna_id.
    changed_by : str
        Who created this record: 'curator', 'llm', or 'system'.
    change_reason : str
        Human-readable reason string stored in the audit trail.

    Returns
    -------
    ContentVersion
        The newly created version row (not yet committed).
    """
    version_record = ContentVersion(
        record_id=qna_record.qna_id,
        record_type="qna",
        pdf_id=qna_record.pdf_id,
        content_snapshot=_build_qna_snapshot(qna_record),
        source_type=qna_record.source_type,
        changed_by=changed_by,
        changed_at=datetime.now(timezone.utc),
        change_reason=change_reason,
        version_number=1,
        is_current=True,
    )
    session.add(version_record)
    logger.debug(
        "Initial version written for qna_id=%s pdf_id=%s",
        qna_record.qna_id,
        qna_record.pdf_id,
    )
    return version_record


def write_initial_version_for_comment(
    session: Session,
    comment_record: CommentsStore,
    changed_by: str,
    change_reason: str,
) -> ContentVersion:
    """
    Write version 1 for a newly created comment record.
    Called during ingestion immediately after the comments_store row is flushed.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session. Caller controls commit/rollback.
    comment_record : CommentsStore
        The freshly created comment row with a valid comment_id.
    changed_by : str
        Who created this record: 'curator', 'llm', or 'system'.
    change_reason : str
        Human-readable reason string stored in the audit trail.

    Returns
    -------
    ContentVersion
        The newly created version row (not yet committed).
    """
    version_record = ContentVersion(
        record_id=comment_record.comment_id,
        record_type="comment",
        pdf_id=comment_record.pdf_id,
        content_snapshot=_build_comment_snapshot(comment_record),
        source_type=comment_record.source_type,
        changed_by=changed_by,
        changed_at=datetime.now(timezone.utc),
        change_reason=change_reason,
        version_number=1,
        is_current=True,
    )
    session.add(version_record)
    logger.debug(
        "Initial version written for comment_id=%s pdf_id=%s",
        comment_record.comment_id,
        comment_record.pdf_id,
    )
    return version_record


def write_edit_version_for_qna(
    session: Session,
    qna_record: QnAStore,
    updated_question: str,
    updated_answer: str,
    changed_by: str,
    change_reason: str,
) -> ContentVersion:
    """
    Write a new version before applying an edit to an existing QnA record.

    Steps performed inside this function (all within the caller's session):
      1. Mark all previous content_versions for this record as is_current=False.
      2. Apply the new question/answer values to the live qna_store row.
      3. Write a new content_versions row with the updated snapshot and is_current=True.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session. Caller controls commit/rollback.
    qna_record : QnAStore
        The existing QnA row to be edited.
    updated_question : str
        New question text.
    updated_answer : str
        New answer text.
    changed_by : str
        Who is making this change: 'curator', 'llm', or 'system'.
    change_reason : str
        Human-readable reason stored in the audit trail.

    Returns
    -------
    ContentVersion
        The newly created version row (not yet committed).
    """
    # Step 1: retire all prior versions for this record
    session.query(ContentVersion).filter(
        ContentVersion.record_id == qna_record.qna_id,
        ContentVersion.record_type == "qna",
        ContentVersion.is_current == True,
    ).update({"is_current": False})

    # Step 2: apply edit to the live record
    qna_record.question = updated_question
    qna_record.answer = updated_answer
    qna_record.last_reviewed_at = datetime.now(timezone.utc)

    next_version_number = _get_next_version_number(
        session, qna_record.qna_id, "qna"
    )

    # Step 3: write new version snapshot
    version_record = ContentVersion(
        record_id=qna_record.qna_id,
        record_type="qna",
        pdf_id=qna_record.pdf_id,
        content_snapshot=_build_qna_snapshot(qna_record),
        source_type=qna_record.source_type,
        changed_by=changed_by,
        changed_at=datetime.now(timezone.utc),
        change_reason=change_reason,
        version_number=next_version_number,
        is_current=True,
    )
    session.add(version_record)

    logger.info(
        "Edit version %s written for qna_id=%s changed_by=%s",
        next_version_number,
        qna_record.qna_id,
        changed_by,
    )
    return version_record


def write_edit_version_for_comment(
    session: Session,
    comment_record: CommentsStore,
    updated_header: str,
    updated_comment: str,
    changed_by: str,
    change_reason: str,
) -> ContentVersion:
    """
    Write a new version before applying an edit to an existing comment record.

    Steps performed inside this function (all within the caller's session):
      1. Mark all previous content_versions for this record as is_current=False.
      2. Apply the new header/comment values to the live comments_store row.
      3. Write a new content_versions row with the updated snapshot and is_current=True.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session. Caller controls commit/rollback.
    comment_record : CommentsStore
        The existing comment row to be edited.
    updated_header : str
        New header text.
    updated_comment : str
        New comment text.
    changed_by : str
        Who is making this change: 'curator', 'llm', or 'system'.
    change_reason : str
        Human-readable reason stored in the audit trail.

    Returns
    -------
    ContentVersion
        The newly created version row (not yet committed).
    """
    # Step 1: retire all prior versions for this record
    session.query(ContentVersion).filter(
        ContentVersion.record_id == comment_record.comment_id,
        ContentVersion.record_type == "comment",
        ContentVersion.is_current == True,
    ).update({"is_current": False})

    # Step 2: apply edit to the live record
    comment_record.header = updated_header
    comment_record.comment = updated_comment
    comment_record.last_reviewed_at = datetime.now(timezone.utc)

    next_version_number = _get_next_version_number(
        session, comment_record.comment_id, "comment"
    )

    # Step 3: write new version snapshot
    version_record = ContentVersion(
        record_id=comment_record.comment_id,
        record_type="comment",
        pdf_id=comment_record.pdf_id,
        content_snapshot=_build_comment_snapshot(comment_record),
        source_type=comment_record.source_type,
        changed_by=changed_by,
        changed_at=datetime.now(timezone.utc),
        change_reason=change_reason,
        version_number=next_version_number,
        is_current=True,
    )
    session.add(version_record)

    logger.info(
        "Edit version %s written for comment_id=%s changed_by=%s",
        next_version_number,
        comment_record.comment_id,
        changed_by,
    )
    return version_record


def get_version_history_for_qna(
    session: Session,
    qna_id: int,
) -> list:
    """
    Return all content_versions rows for a given qna_id, ordered oldest first.
    Used by the Review UI to display edit history.
    """
    return (
        session.query(ContentVersion)
        .filter(
            ContentVersion.record_id == qna_id,
            ContentVersion.record_type == "qna",
        )
        .order_by(ContentVersion.version_number.asc())
        .all()
    )


def get_version_history_for_comment(
    session: Session,
    comment_id: int,
) -> list:
    """
    Return all content_versions rows for a given comment_id, ordered oldest first.
    Used by the Review UI to display edit history.
    """
    return (
        session.query(ContentVersion)
        .filter(
            ContentVersion.record_id == comment_id,
            ContentVersion.record_type == "comment",
        )
        .order_by(ContentVersion.version_number.asc())
        .all()
    )
