"""
benchmark/decay_manager.py

Benchmark decay management for the FDA Clinical Pharmacology Pipeline.

The decay cycle:
  decay_review_due = ingestion_date + 90 days (from config)
  Daily check for entries where:
    decay_review_due <= today  AND  decay_review_completed = FALSE
    AND  benchmark_eligible = TRUE

  Three curator options per overdue entry:
    1. Re-approve:            Reset decay_review_due, mark decay_review_completed.
    2. Re-approve with edit:  Write new content_versions entry, then re-approve.
    3. Remove from benchmark: Set benchmark_eligible=FALSE, benchmark_weight=0.0.

  All edits route through db/versioning.py before touching the live record.
  Removal does not delete the record — it only removes it from the pool.

Eligibility rules enforced here:
  benchmark_eligible = TRUE only for:
    human_curated, auto_corrected, auto_reviewed
  auto_generated entries are NEVER eligible, regardless of curator action.

This module does not schedule itself. The pipeline entry point (app.py or
a cron/APScheduler invocation) calls run_daily_decay_check() on startup
and once per day thereafter.
"""

from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from db.connection import get_session
from db.models import CommentsStore, QnAStore
from db.versioning import write_edit_version_for_comment, write_edit_version_for_qna
from config.settings import Settings
from utils.logger import get_module_logger

logger = get_module_logger(__name__)


# ---------------------------------------------------------------------------
# Source types eligible for benchmark pool
# ---------------------------------------------------------------------------

_BENCHMARK_ELIGIBLE_SOURCE_TYPES = frozenset([
    "human_curated",
    "auto_corrected",
    "auto_reviewed",
])


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class DecayReviewItem:
    """
    A single entry surfaced for curator decay review.
    Presented in the Benchmark Library UI decay review queue.
    """

    record_type: str            # 'qna' or 'comment'
    record_id: int              # qna_id or comment_id
    pdf_id: int
    section_type: str
    source_type: str
    benchmark_weight: float
    decay_review_due: date
    days_overdue: int
    content_preview: str        # First 120 chars of question or header


@dataclass
class DecayCheckResult:
    """
    Summary of a decay check run. Returned to the caller and logged.
    """

    check_date: date
    total_overdue_qna: int
    total_overdue_comments: int
    total_removed_from_pool: int    # Entries with is_overdue but no curator action yet
    items: List[DecayReviewItem] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def compute_initial_decay_review_due(
    ingestion_date: datetime,
    decay_review_days: int,
) -> date:
    """
    Compute the initial decay_review_due date from ingestion date.

    Parameters
    ----------
    ingestion_date : datetime
        The ingestion_date field from qna_store or comments_store.
    decay_review_days : int
        From settings.benchmark.decay_review_days (default 90).

    Returns
    -------
    date
        The date on which this entry should first be reviewed.
    """
    return (ingestion_date + timedelta(days=decay_review_days)).date()


def compute_next_decay_review_due(
    current_date: date,
    decay_review_days: int,
) -> date:
    """
    Compute the next decay_review_due date after a re-approval.
    Resets the 90-day clock from the current date.
    """
    return current_date + timedelta(days=decay_review_days)


# ---------------------------------------------------------------------------
# Overdue query functions
# ---------------------------------------------------------------------------


def get_overdue_qna_entries(today: Optional[date] = None) -> List[QnAStore]:
    """
    Return all QnA entries that are overdue for decay review.
    Filters: benchmark_eligible=TRUE, is_current=TRUE,
             decay_review_completed=FALSE, decay_review_due <= today.
    """
    if today is None:
        today = datetime.now(timezone.utc).date()

    with get_session() as session:
        overdue_rows = (
            session.query(QnAStore)
            .filter(
                QnAStore.benchmark_eligible == True,
                QnAStore.is_current == True,
                QnAStore.decay_review_completed == False,
                QnAStore.decay_review_due <= today,
            )
            .order_by(QnAStore.decay_review_due.asc())
            .all()
        )
        # Detach from session for use outside context
        session.expunge_all()
        return overdue_rows


def get_overdue_comment_entries(today: Optional[date] = None) -> List[CommentsStore]:
    """
    Return all comment entries that are overdue for decay review.
    Same filter logic as get_overdue_qna_entries.
    """
    if today is None:
        today = datetime.now(timezone.utc).date()

    with get_session() as session:
        overdue_rows = (
            session.query(CommentsStore)
            .filter(
                CommentsStore.benchmark_eligible == True,
                CommentsStore.is_current == True,
                CommentsStore.decay_review_completed == False,
                CommentsStore.decay_review_due <= today,
            )
            .order_by(CommentsStore.decay_review_due.asc())
            .all()
        )
        session.expunge_all()
        return overdue_rows


# ---------------------------------------------------------------------------
# Daily check entry point
# ---------------------------------------------------------------------------


def run_daily_decay_check(settings: Settings) -> DecayCheckResult:
    """
    Run the daily decay check. Identifies all overdue benchmark entries
    and returns them for display in the Benchmark Library decay queue.

    This function does NOT automatically remove overdue entries —
    that requires explicit curator action via the three options below.
    It only identifies and surfaces them.

    Parameters
    ----------
    settings : Settings
        Not used directly here but passed for consistency with caller pattern.

    Returns
    -------
    DecayCheckResult
        Summary with counts and the list of overdue DecayReviewItem objects.
    """
    today = datetime.now(timezone.utc).date()

    overdue_qna = get_overdue_qna_entries(today)
    overdue_comments = get_overdue_comment_entries(today)

    review_items: List[DecayReviewItem] = []

    for qna_row in overdue_qna:
        days_overdue = (today - qna_row.decay_review_due).days
        review_items.append(
            DecayReviewItem(
                record_type="qna",
                record_id=qna_row.qna_id,
                pdf_id=qna_row.pdf_id,
                section_type=qna_row.section_type or "",
                source_type=qna_row.source_type or "",
                benchmark_weight=qna_row.benchmark_weight or 0.0,
                decay_review_due=qna_row.decay_review_due,
                days_overdue=days_overdue,
                content_preview=(qna_row.question or "")[:120],
            )
        )

    for comment_row in overdue_comments:
        days_overdue = (today - comment_row.decay_review_due).days
        review_items.append(
            DecayReviewItem(
                record_type="comment",
                record_id=comment_row.comment_id,
                pdf_id=comment_row.pdf_id,
                section_type=comment_row.section_type or "",
                source_type=comment_row.source_type or "",
                benchmark_weight=comment_row.benchmark_weight or 0.0,
                decay_review_due=comment_row.decay_review_due,
                days_overdue=days_overdue,
                content_preview=(comment_row.header or "")[:120],
            )
        )

    # Sort: most overdue first
    review_items.sort(key=lambda item: item.days_overdue, reverse=True)

    result = DecayCheckResult(
        check_date=today,
        total_overdue_qna=len(overdue_qna),
        total_overdue_comments=len(overdue_comments),
        total_removed_from_pool=0,
        items=review_items,
    )

    if review_items:
        logger.info(
            "Decay check: %s overdue QnA, %s overdue comments as of %s",
            len(overdue_qna),
            len(overdue_comments),
            today,
        )
    else:
        logger.info("Decay check: no overdue entries as of %s", today)

    return result


# ---------------------------------------------------------------------------
# Curator action: Option 1 — Re-approve
# ---------------------------------------------------------------------------


def reapprove_qna_entry(
    qna_id: int,
    settings: Settings,
) -> None:
    """
    Curator action: re-approve a QnA entry without edits.
    Resets decay_review_due to today + decay_review_days.
    Sets decay_review_completed = TRUE temporarily (will reset to FALSE
    on next cycle when the new decay_review_due is reached).

    Parameters
    ----------
    qna_id : int
        The QnA entry to re-approve.
    settings : Settings
        Provides benchmark.decay_review_days.
    """
    today = datetime.now(timezone.utc).date()
    new_due_date = compute_next_decay_review_due(today, settings.benchmark.decay_review_days)

    with get_session() as session:
        qna_row = session.query(QnAStore).filter(QnAStore.qna_id == qna_id).first()
        if qna_row is None:
            raise RuntimeError(f"QnA entry qna_id={qna_id} not found.")

        qna_row.decay_review_due = new_due_date
        qna_row.decay_review_completed = True
        qna_row.last_reviewed_at = datetime.now(timezone.utc)

    logger.info(
        "QnA re-approved: qna_id=%s new_decay_due=%s",
        qna_id,
        new_due_date,
    )


def reapprove_comment_entry(
    comment_id: int,
    settings: Settings,
) -> None:
    """
    Curator action: re-approve a comment entry without edits.
    Resets decay_review_due and marks reviewed.
    """
    today = datetime.now(timezone.utc).date()
    new_due_date = compute_next_decay_review_due(today, settings.benchmark.decay_review_days)

    with get_session() as session:
        comment_row = session.query(CommentsStore).filter(
            CommentsStore.comment_id == comment_id
        ).first()
        if comment_row is None:
            raise RuntimeError(f"Comment entry comment_id={comment_id} not found.")

        comment_row.decay_review_due = new_due_date
        comment_row.decay_review_completed = True
        comment_row.last_reviewed_at = datetime.now(timezone.utc)

    logger.info(
        "Comment re-approved: comment_id=%s new_decay_due=%s",
        comment_id,
        new_due_date,
    )


# ---------------------------------------------------------------------------
# Curator action: Option 2 — Re-approve with edit
# ---------------------------------------------------------------------------


def reapprove_qna_with_edit(
    qna_id: int,
    updated_question: str,
    updated_answer: str,
    settings: Settings,
    change_reason: str = "Curator decay review edit",
) -> None:
    """
    Curator action: edit a QnA entry and re-approve it.
    Writes a new content_versions record via versioning.py before
    touching the live record. Updates source_type to 'auto_corrected'
    and benchmark_weight accordingly.

    Parameters
    ----------
    qna_id : int
        The QnA entry to edit and re-approve.
    updated_question : str
        New question text.
    updated_answer : str
        New answer text.
    settings : Settings
        Provides decay and weight configuration.
    change_reason : str
        Stored in the content_versions audit trail.
    """
    today = datetime.now(timezone.utc).date()
    new_due_date = compute_next_decay_review_due(today, settings.benchmark.decay_review_days)

    with get_session() as session:
        qna_row = session.query(QnAStore).filter(QnAStore.qna_id == qna_id).first()
        if qna_row is None:
            raise RuntimeError(f"QnA entry qna_id={qna_id} not found.")

        # Write version before edit — versioning.py applies the edit to the row
        write_edit_version_for_qna(
            session=session,
            qna_record=qna_row,
            updated_question=updated_question,
            updated_answer=updated_answer,
            changed_by="curator",
            change_reason=change_reason,
        )

        # Upgrade source_type and weight after curator edit
        qna_row.source_type = "auto_corrected"
        qna_row.benchmark_weight = settings.benchmark.weight_auto_corrected
        qna_row.benchmark_eligible = True
        qna_row.decay_review_due = new_due_date
        qna_row.decay_review_completed = True

    logger.info(
        "QnA re-approved with edit: qna_id=%s source_type=auto_corrected "
        "new_decay_due=%s",
        qna_id,
        new_due_date,
    )


def reapprove_comment_with_edit(
    comment_id: int,
    updated_header: str,
    updated_comment: str,
    settings: Settings,
    change_reason: str = "Curator decay review edit",
) -> None:
    """
    Curator action: edit a comment entry and re-approve it.
    Writes a new content_versions record before touching the live record.
    Upgrades source_type to 'auto_corrected'.
    """
    today = datetime.now(timezone.utc).date()
    new_due_date = compute_next_decay_review_due(today, settings.benchmark.decay_review_days)

    with get_session() as session:
        comment_row = session.query(CommentsStore).filter(
            CommentsStore.comment_id == comment_id
        ).first()
        if comment_row is None:
            raise RuntimeError(f"Comment entry comment_id={comment_id} not found.")

        write_edit_version_for_comment(
            session=session,
            comment_record=comment_row,
            updated_header=updated_header,
            updated_comment=updated_comment,
            changed_by="curator",
            change_reason=change_reason,
        )

        comment_row.source_type = "auto_corrected"
        comment_row.benchmark_weight = settings.benchmark.weight_auto_corrected
        comment_row.benchmark_eligible = True
        comment_row.decay_review_due = new_due_date
        comment_row.decay_review_completed = True

    logger.info(
        "Comment re-approved with edit: comment_id=%s source_type=auto_corrected "
        "new_decay_due=%s",
        comment_id,
        new_due_date,
    )


# ---------------------------------------------------------------------------
# Curator action: Option 3 — Remove from benchmark pool
# ---------------------------------------------------------------------------


def remove_qna_from_benchmark_pool(qna_id: int) -> None:
    """
    Curator action: remove a QnA entry from the benchmark pool.
    Sets benchmark_eligible=FALSE and benchmark_weight=0.0.
    The record is NOT deleted — it remains in qna_store for review purposes.

    Parameters
    ----------
    qna_id : int
        The QnA entry to remove from the pool.
    """
    with get_session() as session:
        qna_row = session.query(QnAStore).filter(QnAStore.qna_id == qna_id).first()
        if qna_row is None:
            raise RuntimeError(f"QnA entry qna_id={qna_id} not found.")

        qna_row.benchmark_eligible = False
        qna_row.benchmark_weight = 0.0
        qna_row.decay_review_completed = True
        qna_row.last_reviewed_at = datetime.now(timezone.utc)

    logger.info(
        "QnA removed from benchmark pool: qna_id=%s", qna_id
    )


def remove_comment_from_benchmark_pool(comment_id: int) -> None:
    """
    Curator action: remove a comment from the benchmark pool.
    Sets benchmark_eligible=FALSE and benchmark_weight=0.0.
    Record is retained in comments_store.
    """
    with get_session() as session:
        comment_row = session.query(CommentsStore).filter(
            CommentsStore.comment_id == comment_id
        ).first()
        if comment_row is None:
            raise RuntimeError(f"Comment entry comment_id={comment_id} not found.")

        comment_row.benchmark_eligible = False
        comment_row.benchmark_weight = 0.0
        comment_row.decay_review_completed = True
        comment_row.last_reviewed_at = datetime.now(timezone.utc)

    logger.info(
        "Comment removed from benchmark pool: comment_id=%s", comment_id
    )


# ---------------------------------------------------------------------------
# Bulk decay seed: sets initial decay dates on newly ingested entries
# ---------------------------------------------------------------------------


def seed_decay_review_dates_for_pdf(
    pdf_id: int,
    settings: Settings,
) -> int:
    """
    Set initial decay_review_due dates on all benchmark-eligible entries
    for a newly committed document.

    Called by the post-commit hook in two_phase_commit.py (via the
    post_commit_hook callback), after update_coverage_map_for_pdf.

    Parameters
    ----------
    pdf_id : int
        The pdf_id of the newly committed document.
    settings : Settings
        Provides benchmark.decay_review_days.

    Returns
    -------
    int
        Number of entries updated.
    """
    decay_days = settings.benchmark.decay_review_days
    updated_count = 0

    with get_session() as session:
        qna_rows = (
            session.query(QnAStore)
            .filter(
                QnAStore.pdf_id == pdf_id,
                QnAStore.benchmark_eligible == True,
                QnAStore.decay_review_due == None,
            )
            .all()
        )
        for qna_row in qna_rows:
            if qna_row.ingestion_date:
                qna_row.decay_review_due = compute_initial_decay_review_due(
                    qna_row.ingestion_date, decay_days
                )
                updated_count += 1

        comment_rows = (
            session.query(CommentsStore)
            .filter(
                CommentsStore.pdf_id == pdf_id,
                CommentsStore.benchmark_eligible == True,
                CommentsStore.decay_review_due == None,
            )
            .all()
        )
        for comment_row in comment_rows:
            if comment_row.ingestion_date:
                comment_row.decay_review_due = compute_initial_decay_review_due(
                    comment_row.ingestion_date, decay_days
                )
                updated_count += 1

    logger.info(
        "Decay dates seeded: pdf_id=%s entries_updated=%s decay_days=%s",
        pdf_id,
        updated_count,
        decay_days,
    )
    return updated_count
