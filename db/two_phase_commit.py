"""
db/two_phase_commit.py

Two-phase commit for the FDA Clinical Pharmacology Pipeline ingestion flow.

Phase 1 — PostgreSQL writes:
  pdf_metadata record
  qna_store records (with content_versions version 1 for each)
  comments_store records (with content_versions version 1 for each)
  If any write fails -> full rollback of Phase 1; ingestion_job retained for retry.

Phase 2 — ChromaDB vector writes:
  qna question and answer embeddings
  comment header and comment embeddings
  knowledge_base embeddings (if applicable)
  If any write fails -> rollback Phase 1; ingestion_job retained for retry.

Both succeed -> finalize:
  Update ingestion_jobs status to 'committed'
  Update benchmark_coverage_map
  Set decay_review_due on all new entries

This module does NOT perform benchmark nomination or coverage map updates
directly — those are handled in Session 5 modules and called from here
as post-commit hooks after both phases succeed.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, List, Optional

from sqlalchemy.orm import Session

from db.connection import get_session
from db.models import (
    BenchmarkCoverageMap,
    CommentsStore,
    IngestionJob,
    PDFMetadata,
    QnAStore,
)
from db.versioning import (
    write_initial_version_for_comment,
    write_initial_version_for_qna,
)
from utils.logger import get_module_logger

if TYPE_CHECKING:
    pass

logger = get_module_logger(__name__)


# ---------------------------------------------------------------------------
# Data transfer objects passed into the commit
# ---------------------------------------------------------------------------


class QnAIngestionRecord:
    """
    Transient container for a single QnA pair ready for two-phase commit.
    Populated by the ingestion pipeline before calling execute_two_phase_commit.
    """

    def __init__(
        self,
        pdf_id: int,
        file_key: str,
        qna_key: str,
        question: str,
        answer: str,
        question_embedding: List[float],
        answer_embedding: List[float],
        source_type: str,
        benchmark_eligible: bool,
        benchmark_weight: float,
        section_type: str,
        chunk_level: int,
        decay_review_due,
    ):
        self.pdf_id = pdf_id
        self.file_key = file_key
        self.qna_key = qna_key
        self.question = question
        self.answer = answer
        self.question_embedding = question_embedding
        self.answer_embedding = answer_embedding
        self.source_type = source_type
        self.benchmark_eligible = benchmark_eligible
        self.benchmark_weight = benchmark_weight
        self.section_type = section_type
        self.chunk_level = chunk_level
        self.decay_review_due = decay_review_due


class CommentIngestionRecord:
    """
    Transient container for a single comment ready for two-phase commit.
    Populated by the ingestion pipeline before calling execute_two_phase_commit.
    """

    def __init__(
        self,
        pdf_id: int,
        file_key: str,
        comment_key: str,
        header: str,
        comment: str,
        header_embedding: List[float],
        comment_embedding: List[float],
        source_type: str,
        benchmark_eligible: bool,
        benchmark_weight: float,
        section_type: str,
        chunk_level: int,
        decay_review_due,
    ):
        self.pdf_id = pdf_id
        self.file_key = file_key
        self.comment_key = comment_key
        self.header = header
        self.comment = comment
        self.header_embedding = header_embedding
        self.comment_embedding = comment_embedding
        self.source_type = source_type
        self.benchmark_eligible = benchmark_eligible
        self.benchmark_weight = benchmark_weight
        self.section_type = section_type
        self.chunk_level = chunk_level
        self.decay_review_due = decay_review_due


class TwoPhaseCommitPayload:
    """
    Full payload for a single document's two-phase commit.
    Passed to execute_two_phase_commit by the ingestion job manager.
    """

    def __init__(
        self,
        job_id: int,
        pdf_metadata_record: PDFMetadata,
        qna_records: List[QnAIngestionRecord],
        comment_records: List[CommentIngestionRecord],
        chroma_collection_name: str,
    ):
        self.job_id = job_id
        self.pdf_metadata_record = pdf_metadata_record
        self.qna_records = qna_records
        self.comment_records = comment_records
        self.chroma_collection_name = chroma_collection_name


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class TwoPhaseCommitResult:
    """
    Result returned by execute_two_phase_commit.
    Callers check success before proceeding.
    """

    def __init__(
        self,
        success: bool,
        pdf_id: Optional[int],
        committed_qna_count: int,
        committed_comment_count: int,
        failure_phase: Optional[str],
        error_message: Optional[str],
    ):
        self.success = success
        self.pdf_id = pdf_id
        self.committed_qna_count = committed_qna_count
        self.committed_comment_count = committed_comment_count
        self.failure_phase = failure_phase
        self.error_message = error_message


# ---------------------------------------------------------------------------
# Phase 1: PostgreSQL writes
# ---------------------------------------------------------------------------


def _execute_phase_one(
    session: Session,
    payload: TwoPhaseCommitPayload,
) -> tuple:
    """
    Write all PostgreSQL records for a document ingestion within a single session.
    Returns (committed_qna_ids, committed_comment_ids) on success.
    Raises on any failure — caller handles rollback.

    Records written:
      - pdf_metadata (already populated in payload, just added to session)
      - qna_store rows
      - content_versions version 1 for each qna row
      - comments_store rows
      - content_versions version 1 for each comment row
    """
    ingestion_timestamp = datetime.now(timezone.utc)

    # --- pdf_metadata ---
    payload.pdf_metadata_record.ingestion_date = ingestion_timestamp
    session.add(payload.pdf_metadata_record)
    session.flush()  # populate pdf_id via RETURNING
    assigned_pdf_id = payload.pdf_metadata_record.pdf_id
    logger.info(
        "Phase 1: pdf_metadata flushed with pdf_id=%s", assigned_pdf_id
    )

    # --- qna_store + initial content_versions ---
    committed_qna_ids = []
    for qna_input in payload.qna_records:
        qna_row = QnAStore(
            pdf_id=assigned_pdf_id,
            file_key=qna_input.file_key,
            qna_key=qna_input.qna_key,
            question=qna_input.question,
            answer=qna_input.answer,
            question_embedding=qna_input.question_embedding,
            answer_embedding=qna_input.answer_embedding,
            source_type=qna_input.source_type,
            benchmark_eligible=qna_input.benchmark_eligible,
            benchmark_weight=qna_input.benchmark_weight,
            section_type=qna_input.section_type,
            chunk_level=qna_input.chunk_level,
            is_current=True,
            ingestion_date=ingestion_timestamp,
            last_reviewed_at=None,
            decay_review_due=qna_input.decay_review_due,
            decay_review_completed=False,
        )
        session.add(qna_row)
        session.flush()

        write_initial_version_for_qna(
            session=session,
            qna_record=qna_row,
            changed_by="system",
            change_reason="Initial ingestion",
        )
        committed_qna_ids.append(qna_row.qna_id)

    logger.info(
        "Phase 1: %s qna_store rows written for pdf_id=%s",
        len(committed_qna_ids),
        assigned_pdf_id,
    )

    # --- comments_store + initial content_versions ---
    committed_comment_ids = []
    for comment_input in payload.comment_records:
        comment_row = CommentsStore(
            pdf_id=assigned_pdf_id,
            file_key=comment_input.file_key,
            comment_key=comment_input.comment_key,
            header=comment_input.header,
            comment=comment_input.comment,
            header_embedding=comment_input.header_embedding,
            comment_embedding=comment_input.comment_embedding,
            source_type=comment_input.source_type,
            benchmark_eligible=comment_input.benchmark_eligible,
            benchmark_weight=comment_input.benchmark_weight,
            section_type=comment_input.section_type,
            chunk_level=comment_input.chunk_level,
            is_current=True,
            ingestion_date=ingestion_timestamp,
            last_reviewed_at=None,
            decay_review_due=comment_input.decay_review_due,
            decay_review_completed=False,
        )
        session.add(comment_row)
        session.flush()

        write_initial_version_for_comment(
            session=session,
            comment_record=comment_row,
            changed_by="system",
            change_reason="Initial ingestion",
        )
        committed_comment_ids.append(comment_row.comment_id)

    logger.info(
        "Phase 1: %s comments_store rows written for pdf_id=%s",
        len(committed_comment_ids),
        assigned_pdf_id,
    )

    return assigned_pdf_id, committed_qna_ids, committed_comment_ids


# ---------------------------------------------------------------------------
# Phase 2: ChromaDB vector writes
# ---------------------------------------------------------------------------


def _execute_phase_two(
    chroma_collection,
    payload: TwoPhaseCommitPayload,
    assigned_pdf_id: int,
) -> None:
    """
    Write all vector embeddings to ChromaDB for a committed document.
    Raises on any failure — caller handles Phase 1 rollback.

    Writes:
      - question and answer embeddings for each qna record
      - header and comment embeddings for each comment record
    """
    qna_ids = []
    qna_embeddings = []
    qna_documents = []
    qna_metadatas = []

    for index, qna_input in enumerate(payload.qna_records):
        # question embedding
        qna_ids.append(f"qna_question_{assigned_pdf_id}_{qna_input.qna_key}")
        qna_embeddings.append(qna_input.question_embedding)
        qna_documents.append(qna_input.question)
        qna_metadatas.append(
            {
                "pdf_id": assigned_pdf_id,
                "file_key": qna_input.file_key,
                "qna_key": qna_input.qna_key,
                "embedding_type": "question",
                "section_type": qna_input.section_type,
                "chunk_level": qna_input.chunk_level,
                "source_type": qna_input.source_type,
            }
        )

        # answer embedding
        qna_ids.append(f"qna_answer_{assigned_pdf_id}_{qna_input.qna_key}")
        qna_embeddings.append(qna_input.answer_embedding)
        qna_documents.append(qna_input.answer)
        qna_metadatas.append(
            {
                "pdf_id": assigned_pdf_id,
                "file_key": qna_input.file_key,
                "qna_key": qna_input.qna_key,
                "embedding_type": "answer",
                "section_type": qna_input.section_type,
                "chunk_level": qna_input.chunk_level,
                "source_type": qna_input.source_type,
            }
        )

    if qna_ids:
        chroma_collection.add(
            ids=qna_ids,
            embeddings=qna_embeddings,
            documents=qna_documents,
            metadatas=qna_metadatas,
        )
        logger.info(
            "Phase 2: %s QnA embeddings written to ChromaDB for pdf_id=%s",
            len(qna_ids),
            assigned_pdf_id,
        )

    comment_ids = []
    comment_embeddings = []
    comment_documents = []
    comment_metadatas = []

    for comment_input in payload.comment_records:
        # header embedding
        comment_ids.append(
            f"comment_header_{assigned_pdf_id}_{comment_input.comment_key}"
        )
        comment_embeddings.append(comment_input.header_embedding)
        comment_documents.append(comment_input.header)
        comment_metadatas.append(
            {
                "pdf_id": assigned_pdf_id,
                "file_key": comment_input.file_key,
                "comment_key": comment_input.comment_key,
                "embedding_type": "header",
                "section_type": comment_input.section_type,
                "chunk_level": comment_input.chunk_level,
                "source_type": comment_input.source_type,
            }
        )

        # comment embedding
        comment_ids.append(
            f"comment_body_{assigned_pdf_id}_{comment_input.comment_key}"
        )
        comment_embeddings.append(comment_input.comment_embedding)
        comment_documents.append(comment_input.comment)
        comment_metadatas.append(
            {
                "pdf_id": assigned_pdf_id,
                "file_key": comment_input.file_key,
                "comment_key": comment_input.comment_key,
                "embedding_type": "comment",
                "section_type": comment_input.section_type,
                "chunk_level": comment_input.chunk_level,
                "source_type": comment_input.source_type,
            }
        )

    if comment_ids:
        chroma_collection.add(
            ids=comment_ids,
            embeddings=comment_embeddings,
            documents=comment_documents,
            metadatas=comment_metadatas,
        )
        logger.info(
            "Phase 2: %s comment embeddings written to ChromaDB for pdf_id=%s",
            len(comment_ids),
            assigned_pdf_id,
        )


# ---------------------------------------------------------------------------
# Finalize: post-commit updates
# ---------------------------------------------------------------------------


def _finalize_ingestion_job(
    session: Session,
    job_id: int,
    pdf_id: int,
    committed_qna_count: int,
    committed_comment_count: int,
) -> None:
    """
    After both phases succeed, update the ingestion_job to 'committed'.
    Called inside the same session used for Phase 1 commit confirmation.
    """
    job_row = session.query(IngestionJob).filter(
        IngestionJob.job_id == job_id
    ).first()

    if job_row is None:
        logger.warning(
            "Finalize: ingestion_job not found for job_id=%s", job_id
        )
        return

    job_row.status = "committed"
    job_row.last_completed_step = "committed"
    job_row.completed_at = datetime.now(timezone.utc)
    job_row.updated_at = datetime.now(timezone.utc)

    logger.info(
        "Ingestion job job_id=%s finalized as committed. "
        "qna_count=%s comment_count=%s pdf_id=%s",
        job_id,
        committed_qna_count,
        committed_comment_count,
        pdf_id,
    )


def _mark_job_failed(
    job_id: int,
    failure_phase: str,
    error_message: str,
) -> None:
    """
    Update the ingestion_job status to 'failed' with error detail.
    Uses its own session so it can write even after a Phase 1 rollback.
    """
    try:
        with get_session() as failure_session:
            job_row = failure_session.query(IngestionJob).filter(
                IngestionJob.job_id == job_id
            ).first()
            if job_row is not None:
                job_row.status = "failed"
                job_row.error_message = (
                    f"[{failure_phase}] {error_message}"
                )
                job_row.updated_at = datetime.now(timezone.utc)
                job_row.retry_count = (job_row.retry_count or 0) + 1
                logger.error(
                    "Job job_id=%s marked failed at %s: %s",
                    job_id,
                    failure_phase,
                    error_message,
                )
    except Exception as status_write_error:
        logger.error(
            "Could not update job status to failed for job_id=%s: %s",
            job_id,
            status_write_error,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute_two_phase_commit(
    payload: TwoPhaseCommitPayload,
    chroma_collection,
    post_commit_hook: Optional[Callable[[int], None]] = None,
) -> TwoPhaseCommitResult:
    """
    Execute the full two-phase commit for one document ingestion batch.

    Phase 1 writes all PostgreSQL records within a single transaction.
    Phase 2 writes all ChromaDB vectors.
    Any failure in either phase triggers Phase 1 rollback and job failure logging.

    Parameters
    ----------
    payload : TwoPhaseCommitPayload
        All records and metadata for the document being committed.
    chroma_collection
        Initialized ChromaDB collection object (from vector_store/index_manager.py).
    post_commit_hook : Callable[[int], None], optional
        Optional callback invoked after both phases succeed.
        Receives pdf_id. Used to trigger nomination_engine and coverage_map updates.

    Returns
    -------
    TwoPhaseCommitResult
        Describes success/failure and counts of committed records.
    """
    assigned_pdf_id: Optional[int] = None
    committed_qna_ids: list = []
    committed_comment_ids: list = []

    # ------------------------------------------------------------------
    # Phase 1: PostgreSQL
    # ------------------------------------------------------------------
    try:
        with get_session() as phase_one_session:
            assigned_pdf_id, committed_qna_ids, committed_comment_ids = (
                _execute_phase_one(phase_one_session, payload)
            )
            # Phase 1 session commits on clean exit of context manager

        logger.info(
            "Phase 1 committed successfully for job_id=%s pdf_id=%s",
            payload.job_id,
            assigned_pdf_id,
        )

    except Exception as phase_one_error:
        error_text = str(phase_one_error)
        logger.error(
            "Phase 1 failed for job_id=%s — rolling back. Error: %s",
            payload.job_id,
            error_text,
        )
        _mark_job_failed(
            job_id=payload.job_id,
            failure_phase="Phase 1 PostgreSQL",
            error_message=error_text,
        )
        return TwoPhaseCommitResult(
            success=False,
            pdf_id=None,
            committed_qna_count=0,
            committed_comment_count=0,
            failure_phase="Phase 1 PostgreSQL",
            error_message=error_text,
        )

    # ------------------------------------------------------------------
    # Phase 2: ChromaDB
    # ------------------------------------------------------------------
    try:
        _execute_phase_two(chroma_collection, payload, assigned_pdf_id)
        logger.info(
            "Phase 2 committed successfully for job_id=%s pdf_id=%s",
            payload.job_id,
            assigned_pdf_id,
        )

    except Exception as phase_two_error:
        error_text = str(phase_two_error)
        logger.error(
            "Phase 2 failed for job_id=%s — rolling back Phase 1. Error: %s",
            payload.job_id,
            error_text,
        )

        # Roll back Phase 1: delete the PostgreSQL records just written
        try:
            with get_session() as rollback_session:
                rollback_session.query(QnAStore).filter(
                    QnAStore.pdf_id == assigned_pdf_id
                ).delete()
                rollback_session.query(CommentsStore).filter(
                    CommentsStore.pdf_id == assigned_pdf_id
                ).delete()
                rollback_session.query(PDFMetadata).filter(
                    PDFMetadata.pdf_id == assigned_pdf_id
                ).delete()
                logger.info(
                    "Phase 1 rollback completed for pdf_id=%s", assigned_pdf_id
                )
        except Exception as rollback_error:
            logger.error(
                "Phase 1 rollback also failed for pdf_id=%s: %s",
                assigned_pdf_id,
                rollback_error,
            )

        _mark_job_failed(
            job_id=payload.job_id,
            failure_phase="Phase 2 ChromaDB",
            error_message=error_text,
        )
        return TwoPhaseCommitResult(
            success=False,
            pdf_id=assigned_pdf_id,
            committed_qna_count=0,
            committed_comment_count=0,
            failure_phase="Phase 2 ChromaDB",
            error_message=error_text,
        )

    # ------------------------------------------------------------------
    # Finalize: mark job committed, run post-commit hook
    # ------------------------------------------------------------------
    try:
        with get_session() as finalize_session:
            _finalize_ingestion_job(
                session=finalize_session,
                job_id=payload.job_id,
                pdf_id=assigned_pdf_id,
                committed_qna_count=len(committed_qna_ids),
                committed_comment_count=len(committed_comment_ids),
            )

        if post_commit_hook is not None:
            try:
                post_commit_hook(assigned_pdf_id)
            except Exception as hook_error:
                # Post-commit hooks are non-fatal — both phases already succeeded
                logger.warning(
                    "Post-commit hook raised a non-fatal error for pdf_id=%s: %s",
                    assigned_pdf_id,
                    hook_error,
                )

    except Exception as finalize_error:
        # Finalization failure is logged but not treated as a commit failure
        # because both phases already succeeded
        logger.error(
            "Finalization step failed for job_id=%s pdf_id=%s: %s",
            payload.job_id,
            assigned_pdf_id,
            finalize_error,
        )

    return TwoPhaseCommitResult(
        success=True,
        pdf_id=assigned_pdf_id,
        committed_qna_count=len(committed_qna_ids),
        committed_comment_count=len(committed_comment_ids),
        failure_phase=None,
        error_message=None,
    )
