"""
quality/coverage_map.py

Benchmark coverage map maintenance for the FDA Clinical Pharmacology Pipeline.

Responsibilities:
  - Recompute benchmark_coverage_map entries after each commit.
  - Count benchmark-eligible entries by drug class and section type.
  - Assign coverage_status based on benchmark_minimum_coverage from config.
  - Set adjusted_accept_threshold from the spec-defined mapping:
      sufficient (>= 3 entries)  ->  0.85  (quality_score_auto_accept)
      partial    (1–2 entries)   ->  0.70
      none       (0 entries)     ->  0.55

Called as a post-commit hook from db/two_phase_commit.py after both
phases succeed. Also callable on demand from the Benchmark Library UI
to force a refresh.

Drug class derivation:
  The pipeline derives drug class from drug_name_generic using a small
  static lookup. For drugs not in the lookup, the drug name itself is
  used as its own class (one-entry class). This is intentionally simple —
  the curator can refine entries via the Benchmark Library UI.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from db.connection import get_session
from db.models import BenchmarkCoverageMap, CommentsStore, PDFMetadata, QnAStore
from config.settings import Settings
from utils.logger import get_module_logger

logger = get_module_logger(__name__)


# ---------------------------------------------------------------------------
# Coverage status constants
# ---------------------------------------------------------------------------

COVERAGE_STATUS_SUFFICIENT = "sufficient"
COVERAGE_STATUS_PARTIAL = "partial"
COVERAGE_STATUS_NONE = "none"

# Adjusted thresholds from the spec
THRESHOLD_SUFFICIENT = 0.85
THRESHOLD_PARTIAL = 0.70
THRESHOLD_NONE = 0.55


# ---------------------------------------------------------------------------
# Drug class lookup
# ---------------------------------------------------------------------------

# Partial, extensible lookup: generic name fragment -> drug class.
# Matching is case-insensitive substring match.
# Curator can extend by editing entries in the Benchmark Library UI.
_DRUG_CLASS_LOOKUP: List[Tuple[str, str]] = [
    # Anticoagulants
    ("warfarin", "anticoagulant"),
    ("rivaroxaban", "anticoagulant"),
    ("apixaban", "anticoagulant"),
    ("dabigatran", "anticoagulant"),
    ("edoxaban", "anticoagulant"),

    # Kinase inhibitors
    ("tinib", "kinase_inhibitor"),      # suffix match for -tinib drugs
    ("rafenib", "kinase_inhibitor"),

    # Monoclonal antibodies
    ("mab", "monoclonal_antibody"),     # suffix match for -mab drugs

    # Statins
    ("statin", "statin"),
    ("vastatin", "statin"),

    # Proton pump inhibitors
    ("prazole", "proton_pump_inhibitor"),

    # Antibiotics
    ("cillin", "antibiotic_penicillin"),
    ("mycin", "antibiotic_macrolide"),
    ("floxacin", "antibiotic_fluoroquinolone"),

    # Antifungals
    ("azole", "antifungal_azole"),
    ("conazole", "antifungal_azole"),

    # Immunosuppressants
    ("cyclosporine", "immunosuppressant"),
    ("tacrolimus", "immunosuppressant"),
    ("sirolimus", "immunosuppressant"),

    # HIV antivirals
    ("navir", "hiv_protease_inhibitor"),
    ("tegravir", "hiv_integrase_inhibitor"),
    ("vudine", "nucleoside_reverse_transcriptase_inhibitor"),

    # Antidiabetics
    ("gliptin", "dpp4_inhibitor"),
    ("gliflozin", "sglt2_inhibitor"),
    ("metformin", "biguanide"),

    # Antipsychotics
    ("piperidol", "antipsychotic"),
    ("peridol", "antipsychotic"),
    ("olanzapine", "antipsychotic"),
    ("quetiapine", "antipsychotic"),
    ("risperidone", "antipsychotic"),
]


def derive_drug_class(drug_name_generic: Optional[str]) -> str:
    """
    Derive a drug class label from a generic drug name.

    Uses substring matching against the _DRUG_CLASS_LOOKUP table.
    Returns the drug name itself (lowercased) if no class is found,
    so every drug gets its own coverage map entry as a fallback.

    Parameters
    ----------
    drug_name_generic : str or None
        The generic drug name from pdf_metadata.

    Returns
    -------
    str
        Drug class string used as the drug_class key in benchmark_coverage_map.
    """
    if not drug_name_generic:
        return "unknown"

    name_lower = drug_name_generic.lower().strip()

    for fragment, drug_class in _DRUG_CLASS_LOOKUP:
        if fragment in name_lower:
            return drug_class

    # No match — use the drug name itself as a single-drug class
    return name_lower


# ---------------------------------------------------------------------------
# Count helpers
# ---------------------------------------------------------------------------


def _count_eligible_qna_by_section(
    drug_class: str,
    section_type: str,
) -> int:
    """
    Count benchmark-eligible QnA entries for a drug class and section type.
    Joins to pdf_metadata to resolve drug_class via derive_drug_class.
    """
    with get_session() as session:
        all_qna_rows = (
            session.query(QnAStore, PDFMetadata)
            .join(PDFMetadata, QnAStore.pdf_id == PDFMetadata.pdf_id)
            .filter(
                QnAStore.benchmark_eligible == True,
                QnAStore.is_current == True,
                QnAStore.section_type == section_type,
            )
            .all()
        )

        count = sum(
            1
            for qna_row, pdf_row in all_qna_rows
            if derive_drug_class(pdf_row.drug_name_generic) == drug_class
        )
    return count


def _count_eligible_comments_by_section(
    drug_class: str,
    section_type: str,
) -> int:
    """
    Count benchmark-eligible comment entries for a drug class and section type.
    """
    with get_session() as session:
        all_comment_rows = (
            session.query(CommentsStore, PDFMetadata)
            .join(PDFMetadata, CommentsStore.pdf_id == PDFMetadata.pdf_id)
            .filter(
                CommentsStore.benchmark_eligible == True,
                CommentsStore.is_current == True,
                CommentsStore.section_type == section_type,
            )
            .all()
        )

        count = sum(
            1
            for comment_row, pdf_row in all_comment_rows
            if derive_drug_class(pdf_row.drug_name_generic) == drug_class
        )
    return count


# ---------------------------------------------------------------------------
# Status and threshold computation
# ---------------------------------------------------------------------------


def compute_coverage_status_and_threshold(
    curated_entry_count: int,
    settings: Settings,
) -> Tuple[str, float]:
    """
    Compute coverage_status and adjusted_accept_threshold from entry count.

    Thresholds from the spec:
      sufficient (>= benchmark_minimum_coverage)  ->  quality_score_auto_accept (0.85)
      partial    (1 to minimum - 1)               ->  0.70
      none       (0)                              ->  0.55

    Parameters
    ----------
    curated_entry_count : int
        Total benchmark-eligible entries for this drug class + section type.
    settings : Settings
        Provides benchmark_minimum_coverage and quality_score_auto_accept.

    Returns
    -------
    Tuple[str, float]
        (coverage_status, adjusted_accept_threshold)
    """
    minimum_coverage = settings.pipeline.benchmark_minimum_coverage

    if curated_entry_count >= minimum_coverage:
        return COVERAGE_STATUS_SUFFICIENT, THRESHOLD_SUFFICIENT
    elif curated_entry_count >= 1:
        return COVERAGE_STATUS_PARTIAL, THRESHOLD_PARTIAL
    else:
        return COVERAGE_STATUS_NONE, THRESHOLD_NONE


# ---------------------------------------------------------------------------
# Upsert helper
# ---------------------------------------------------------------------------


def _upsert_coverage_map_row(
    session,
    drug_class: str,
    section_type: str,
    curated_entry_count: int,
    coverage_status: str,
    adjusted_accept_threshold: float,
) -> None:
    """
    Insert or update a benchmark_coverage_map row.
    Uses drug_class + section_type as the natural key.
    """
    now = datetime.now(timezone.utc)

    existing_row = (
        session.query(BenchmarkCoverageMap)
        .filter(
            BenchmarkCoverageMap.drug_class == drug_class,
            BenchmarkCoverageMap.section_type == section_type,
        )
        .first()
    )

    if existing_row is not None:
        existing_row.curated_entry_count = curated_entry_count
        existing_row.coverage_status = coverage_status
        existing_row.adjusted_accept_threshold = adjusted_accept_threshold
        existing_row.last_updated = now
    else:
        new_row = BenchmarkCoverageMap(
            drug_class=drug_class,
            section_type=section_type,
            curated_entry_count=curated_entry_count,
            last_updated=now,
            coverage_status=coverage_status,
            adjusted_accept_threshold=adjusted_accept_threshold,
        )
        session.add(new_row)


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------


def update_coverage_map_for_pdf(
    pdf_id: int,
    settings: Settings,
) -> int:
    """
    Update benchmark_coverage_map for all section types present in a newly
    committed document.

    Called as a post-commit hook from two_phase_commit.py after both
    Phase 1 and Phase 2 succeed.

    Workflow:
      1. Determine drug_class from the committed document's drug_name_generic.
      2. Identify all section types that received new QnA/comment entries.
      3. Recount all benchmark-eligible entries per section type for that class.
      4. Upsert benchmark_coverage_map rows.

    Parameters
    ----------
    pdf_id : int
        The pdf_id of the just-committed document.
    settings : Settings
        Provides benchmark_minimum_coverage and thresholds.

    Returns
    -------
    int
        Number of coverage map rows updated or inserted.
    """
    # Fetch the drug name for this pdf
    with get_session() as session:
        pdf_row = session.query(PDFMetadata).filter(
            PDFMetadata.pdf_id == pdf_id
        ).first()

        if pdf_row is None:
            logger.warning(
                "coverage_map update: pdf_id=%s not found — skipping.", pdf_id
            )
            return 0

        drug_name_generic = pdf_row.drug_name_generic
        drug_class = derive_drug_class(drug_name_generic)

        # Collect all section types in the new document's QnA and comment entries
        qna_section_types = [
            row[0]
            for row in session.query(QnAStore.section_type)
            .filter(QnAStore.pdf_id == pdf_id)
            .distinct()
            .all()
            if row[0]
        ]

        comment_section_types = [
            row[0]
            for row in session.query(CommentsStore.section_type)
            .filter(CommentsStore.pdf_id == pdf_id)
            .distinct()
            .all()
            if row[0]
        ]

    all_section_types = list(set(qna_section_types + comment_section_types))

    if not all_section_types:
        logger.info(
            "coverage_map update: no sections found for pdf_id=%s — skipping.", pdf_id
        )
        return 0

    updated_count = 0

    with get_session() as session:
        for section_type in all_section_types:
            qna_count = _count_eligible_qna_by_section(drug_class, section_type)
            comment_count = _count_eligible_comments_by_section(drug_class, section_type)
            total_count = qna_count + comment_count

            coverage_status, adjusted_threshold = compute_coverage_status_and_threshold(
                curated_entry_count=total_count,
                settings=settings,
            )

            _upsert_coverage_map_row(
                session=session,
                drug_class=drug_class,
                section_type=section_type,
                curated_entry_count=total_count,
                coverage_status=coverage_status,
                adjusted_accept_threshold=adjusted_threshold,
            )
            updated_count += 1

            logger.debug(
                "Coverage map updated: drug_class=%s section=%s count=%s "
                "status=%s threshold=%.2f",
                drug_class,
                section_type,
                total_count,
                coverage_status,
                adjusted_threshold,
            )

    logger.info(
        "Coverage map update complete: pdf_id=%s drug_class=%s "
        "sections_updated=%s",
        pdf_id,
        drug_class,
        updated_count,
    )
    return updated_count


def refresh_full_coverage_map(settings: Settings) -> int:
    """
    Recompute all benchmark_coverage_map entries from scratch.

    Called from the Benchmark Library UI or on demand after bulk
    curator operations that change benchmark_eligible flags across
    many records.

    Returns
    -------
    int
        Total number of coverage map rows written.
    """
    logger.info("Full coverage map refresh started.")

    # Enumerate all (drug_class, section_type) combinations in qna_store
    with get_session() as session:
        qna_pairs = (
            session.query(QnAStore.section_type, PDFMetadata.drug_name_generic)
            .join(PDFMetadata, QnAStore.pdf_id == PDFMetadata.pdf_id)
            .filter(QnAStore.benchmark_eligible == True, QnAStore.is_current == True)
            .distinct()
            .all()
        )

        comment_pairs = (
            session.query(CommentsStore.section_type, PDFMetadata.drug_name_generic)
            .join(PDFMetadata, CommentsStore.pdf_id == PDFMetadata.pdf_id)
            .filter(
                CommentsStore.benchmark_eligible == True,
                CommentsStore.is_current == True,
            )
            .distinct()
            .all()
        )

    all_pairs = set()
    for section_type, drug_name in qna_pairs:
        if section_type:
            all_pairs.add((derive_drug_class(drug_name), section_type))
    for section_type, drug_name in comment_pairs:
        if section_type:
            all_pairs.add((derive_drug_class(drug_name), section_type))

    updated_count = 0

    with get_session() as session:
        for drug_class, section_type in all_pairs:
            qna_count = _count_eligible_qna_by_section(drug_class, section_type)
            comment_count = _count_eligible_comments_by_section(drug_class, section_type)
            total_count = qna_count + comment_count

            coverage_status, adjusted_threshold = compute_coverage_status_and_threshold(
                curated_entry_count=total_count,
                settings=settings,
            )

            _upsert_coverage_map_row(
                session=session,
                drug_class=drug_class,
                section_type=section_type,
                curated_entry_count=total_count,
                coverage_status=coverage_status,
                adjusted_accept_threshold=adjusted_threshold,
            )
            updated_count += 1

    logger.info("Full coverage map refresh complete: %s rows written.", updated_count)
    return updated_count


def get_coverage_summary() -> List[dict]:
    """
    Return all coverage map rows as a list of dicts for the Benchmark Library UI.

    Returns
    -------
    List[dict]
        Each dict has: drug_class, section_type, curated_entry_count,
        coverage_status, adjusted_accept_threshold, last_updated.
    """
    with get_session() as session:
        rows = (
            session.query(BenchmarkCoverageMap)
            .order_by(
                BenchmarkCoverageMap.drug_class,
                BenchmarkCoverageMap.section_type,
            )
            .all()
        )

        return [
            {
                "drug_class": row.drug_class,
                "section_type": row.section_type,
                "curated_entry_count": row.curated_entry_count,
                "coverage_status": row.coverage_status,
                "adjusted_accept_threshold": row.adjusted_accept_threshold,
                "last_updated": row.last_updated,
            }
            for row in rows
        ]


def get_coverage_gaps() -> List[dict]:
    """
    Return coverage map rows where coverage_status is 'none' or 'partial'.
    Used by the Pipeline Status UI to display drug class coverage gaps.

    Returns
    -------
    List[dict]
        Ordered by coverage_status (none first), then section_type.
    """
    with get_session() as session:
        rows = (
            session.query(BenchmarkCoverageMap)
            .filter(
                BenchmarkCoverageMap.coverage_status.in_(
                    [COVERAGE_STATUS_NONE, COVERAGE_STATUS_PARTIAL]
                )
            )
            .order_by(
                BenchmarkCoverageMap.coverage_status,
                BenchmarkCoverageMap.section_type,
            )
            .all()
        )

        return [
            {
                "drug_class": row.drug_class,
                "section_type": row.section_type,
                "curated_entry_count": row.curated_entry_count,
                "coverage_status": row.coverage_status,
                "adjusted_accept_threshold": row.adjusted_accept_threshold,
            }
            for row in rows
        ]
