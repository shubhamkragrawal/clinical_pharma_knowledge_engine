"""
db/models.py

SQLAlchemy ORM models for the FDA Clinical Pharmacology Pipeline.
Defines all 9 tables exactly as specified in the handoff document.

Tables:
  1. pdf_metadata
  2. qna_store
  3. comments_store
  4. content_versions
  5. ingestion_jobs
  6. knowledge_base
  7. knowledge_base_nominations
  8. source_log
  9. benchmark_coverage_map

pgvector is required for VECTOR columns.
Run: CREATE EXTENSION IF NOT EXISTS vector; in PostgreSQL before creating tables.
"""

from datetime import date, datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    Date,
    Float,
    Integer,
    String,
    Text,
    TIMESTAMP,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Table 1: pdf_metadata
# ---------------------------------------------------------------------------


class PDFMetadata(Base):
    """
    Master record for each ingested FDA Clinical Pharmacology PDF.
    One row per uploaded document.
    """

    __tablename__ = "pdf_metadata"

    pdf_id = Column(Integer, primary_key=True, autoincrement=True)
    file_key = Column(Text)
    file_name = Column(Text)
    drug_name_brand = Column(Text)
    drug_name_generic = Column(Text)
    formulation = Column(Text)
    nda_bla_number = Column(Text)
    applicant_company = Column(Text)
    approval_date = Column(Date)
    submission_date = Column(Date)
    review_type = Column(Text)
    indication = Column(ARRAY(Text))
    review_division = Column(Text)
    document_type = Column(Text)
    source_url = Column(Text)
    pdf_file_path = Column(Text)
    companion_nda = Column(Text)
    ingestion_date = Column(TIMESTAMP)
    has_qna = Column(Boolean)
    has_comments = Column(Boolean)
    curator_notes = Column(Text)


# ---------------------------------------------------------------------------
# Table 2: qna_store
# ---------------------------------------------------------------------------


class QnAStore(Base):
    """
    Question-and-answer pairs generated or curated from PDF sections.
    source_type controls benchmark eligibility and weight.

    Valid source_type values:
      'human_curated', 'auto_generated', 'auto_reviewed', 'auto_corrected'
    """

    __tablename__ = "qna_store"

    qna_id = Column(Integer, primary_key=True, autoincrement=True)
    pdf_id = Column(Integer)                          # FK -> pdf_metadata.pdf_id
    file_key = Column(Text)
    qna_key = Column(Text)
    question = Column(Text)
    answer = Column(Text)
    question_embedding = Column(Vector(768))
    answer_embedding = Column(Vector(768))
    source_type = Column(Text)
    benchmark_eligible = Column(Boolean)
    benchmark_weight = Column(Float)
    section_type = Column(Text)
    chunk_level = Column(Integer)
    is_current = Column(Boolean)
    ingestion_date = Column(TIMESTAMP)
    last_reviewed_at = Column(TIMESTAMP)
    decay_review_due = Column(Date)
    decay_review_completed = Column(Boolean, default=False)


# ---------------------------------------------------------------------------
# Table 3: comments_store
# ---------------------------------------------------------------------------


class CommentsStore(Base):
    """
    Reviewer comments generated or curated from PDF sections.
    source_type controls benchmark eligibility and weight.

    Valid source_type values:
      'human_curated', 'auto_generated', 'auto_reviewed', 'auto_corrected'
    """

    __tablename__ = "comments_store"

    comment_id = Column(Integer, primary_key=True, autoincrement=True)
    pdf_id = Column(Integer)                          # FK -> pdf_metadata.pdf_id
    file_key = Column(Text)
    comment_key = Column(Text)
    header = Column(Text)
    comment = Column(Text)
    header_embedding = Column(Vector(768))
    comment_embedding = Column(Vector(768))
    source_type = Column(Text)
    benchmark_eligible = Column(Boolean)
    benchmark_weight = Column(Float)
    section_type = Column(Text)
    chunk_level = Column(Integer)
    is_current = Column(Boolean)
    ingestion_date = Column(TIMESTAMP)
    last_reviewed_at = Column(TIMESTAMP)
    decay_review_due = Column(Date)
    decay_review_completed = Column(Boolean, default=False)


# ---------------------------------------------------------------------------
# Table 4: content_versions
# ---------------------------------------------------------------------------


class ContentVersion(Base):
    """
    Immutable audit trail for every edit to a QnA or comment record.
    Every change writes a new row here before touching qna_store or comments_store.

    content_snapshot JSONB shape:
      for qna:     {"question": "...", "answer": "..."}
      for comment: {"header": "...", "comment": "..."}

    Valid record_type values:   'qna', 'comment'
    Valid changed_by values:    'curator', 'llm', 'system'
    """

    __tablename__ = "content_versions"

    version_id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(Integer)
    record_type = Column(Text)
    pdf_id = Column(Integer)                          # FK -> pdf_metadata.pdf_id
    content_snapshot = Column(JSONB)
    source_type = Column(Text)
    changed_by = Column(Text)
    changed_at = Column(TIMESTAMP)
    change_reason = Column(Text)
    version_number = Column(Integer)
    is_current = Column(Boolean)


# ---------------------------------------------------------------------------
# Table 5: ingestion_jobs
# ---------------------------------------------------------------------------


class IngestionJob(Base):
    """
    Tracks the state of every PDF ingestion run.
    Used for resume-on-failure logic and the Pipeline Status UI.

    Valid status values:
      'started', 'extracting', 'chunking', 'metadata_extracted',
      'benchmark_retrieved', 'generating_qna', 'generating_comments',
      'quality_scoring', 'review_pending', 'committed', 'failed'
    """

    __tablename__ = "ingestion_jobs"

    job_id = Column(Integer, primary_key=True, autoincrement=True)
    pdf_id = Column(Integer)                          # FK -> pdf_metadata.pdf_id
    file_name = Column(Text)
    status = Column(Text)
    last_completed_step = Column(Text)
    sections_completed = Column(ARRAY(Text))
    sections_total = Column(Integer)
    started_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    llm_mode_used = Column(Text)


# ---------------------------------------------------------------------------
# Table 6: knowledge_base
# ---------------------------------------------------------------------------


class KnowledgeBase(Base):
    """
    Curated, promoted knowledge entries available for query-time retrieval.

    Valid source_type values:
      'qna_promoted', 'comment_promoted', 'online_promoted', 'manual_curated'

    Valid promoted_by values:
      'curator', 'system_suggested'
    """

    __tablename__ = "knowledge_base"

    kb_id = Column(Integer, primary_key=True, autoincrement=True)
    drug_name_generic = Column(Text)
    drug_name_brand = Column(Text)
    formulation = Column(Text)
    nda_bla_number = Column(Text)
    indication = Column(ARRAY(Text))
    content = Column(Text)
    source_type = Column(Text)
    source_record_ids = Column(ARRAY(Integer))
    source_pdf_ids = Column(ARRAY(Integer))
    promoted_by = Column(Text)
    promotion_confirmed_at = Column(TIMESTAMP)
    verified = Column(Boolean)
    content_embedding = Column(Vector(768))
    created_at = Column(TIMESTAMP)
    is_current = Column(Boolean)


# ---------------------------------------------------------------------------
# Table 7: knowledge_base_nominations
# ---------------------------------------------------------------------------


class KnowledgeBaseNomination(Base):
    """
    System-suggested candidates for promotion to the knowledge_base.
    Curator reviews each nomination and decides: promoted, rejected, or deferred.

    Valid record_type values:     'qna', 'comment'
    Valid curator_decision values: 'promoted', 'rejected', 'deferred'
    """

    __tablename__ = "knowledge_base_nominations"

    nomination_id = Column(Integer, primary_key=True, autoincrement=True)
    record_type = Column(Text)
    record_ids = Column(ARRAY(Integer))
    pdf_ids = Column(ARRAY(Integer))
    drug_name_generic = Column(Text)
    section_type = Column(Text)
    content_summary = Column(Text)
    agreement_score = Column(Float)
    nominated_at = Column(TIMESTAMP)
    reviewed_by_curator = Column(Boolean, default=False)
    curator_decision = Column(Text)
    decided_at = Column(TIMESTAMP)


# ---------------------------------------------------------------------------
# Table 8: source_log
# ---------------------------------------------------------------------------


class SourceLog(Base):
    """
    Audit log for every online source retrieval (Drugs@FDA, DailyMed, PubMed, scraping).
    Tracks whether the result was verified and promoted to the knowledge_base.

    kb_id is nullable — only populated after promotion.
    """

    __tablename__ = "source_log"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    query_session_id = Column(Text)
    source_name = Column(Text)
    url = Column(Text)
    retrieval_timestamp = Column(TIMESTAMP)
    response_status = Column(Integer)
    result_count = Column(Integer)
    verified = Column(Boolean, default=False)
    promoted_to_kb = Column(Boolean, default=False)
    promoted_at = Column(TIMESTAMP)
    promoted_by = Column(Text)
    kb_id = Column(Integer)                           # FK -> knowledge_base.kb_id (nullable)


# ---------------------------------------------------------------------------
# Table 9: benchmark_coverage_map
# ---------------------------------------------------------------------------


class BenchmarkCoverageMap(Base):
    """
    Tracks benchmark coverage by drug class and section type.
    Controls adaptive auto-accept thresholds at ingestion time.

    Valid coverage_status values: 'sufficient', 'partial', 'none'

    adjusted_accept_threshold mapping (from spec):
      sufficient  -> 0.85
      partial     -> 0.70
      none        -> 0.55
    """

    __tablename__ = "benchmark_coverage_map"

    coverage_id = Column(Integer, primary_key=True, autoincrement=True)
    drug_class = Column(Text)
    section_type = Column(Text)
    curated_entry_count = Column(Integer)
    last_updated = Column(TIMESTAMP)
    coverage_status = Column(Text)
    adjusted_accept_threshold = Column(Float)
