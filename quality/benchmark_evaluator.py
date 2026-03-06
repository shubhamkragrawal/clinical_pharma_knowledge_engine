"""
quality/benchmark_evaluator.py

Benchmark-based quality evaluation for the FDA Clinical Pharmacology Pipeline.

Evaluates LLM-generated QnA and comment outputs on two dimensions:

  Depth score:
    Measures analytical interpretation vs mere summary.
    High-depth responses: interpret findings, cite specific values,
    draw clinical or regulatory conclusions, note study limitations.
    Low-depth responses: restate the text, omit numerical context,
    make no interpretive claims.

  Coverage score:
    Checks for regulatory flags, study limitations, cross-document context.
    Key elements: labeling implications, dose adjustment recommendations,
    special population considerations, DDI clinical significance,
    study design limitations.

Combined quality score → auto-accept, manual review, or auto-reject decision:
  score >= adjusted_accept_threshold  →  AUTO_ACCEPT
  score <= quality_score_auto_reject  →  AUTO_REJECT
  otherwise                           →  REVIEW_PENDING

source_type is assigned based on the decision:
  AUTO_ACCEPT  →  'auto_reviewed'
  AUTO_REJECT  →  record flagged, not committed
  REVIEW       →  'auto_generated' until curator acts

benchmark_weight is assigned based on source_type:
  human_curated   1.0
  auto_corrected  0.8
  auto_reviewed   0.6
  auto_generated  0.0

This module does not write to the database. Callers receive an
EvaluationResult and act on it.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import Settings
from ingestion.benchmark_retriever import BenchmarkContext
from utils.logger import get_module_logger

logger = get_module_logger(__name__)


# ---------------------------------------------------------------------------
# Decision constants
# ---------------------------------------------------------------------------

DECISION_AUTO_ACCEPT = "auto_accept"
DECISION_REVIEW_PENDING = "review_pending"
DECISION_AUTO_REJECT = "auto_reject"

SOURCE_TYPE_HUMAN_CURATED = "human_curated"
SOURCE_TYPE_AUTO_CORRECTED = "auto_corrected"
SOURCE_TYPE_AUTO_REVIEWED = "auto_reviewed"
SOURCE_TYPE_AUTO_GENERATED = "auto_generated"


# ---------------------------------------------------------------------------
# Depth scoring signals
# ---------------------------------------------------------------------------

# Patterns indicating analytical depth rather than summary
_DEPTH_POSITIVE_PATTERNS = [
    # Numerical values with clinical context
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|mL|g|%|fold|x)\b.*?(?:increase|decrease|"
        r"higher|lower|greater|less|compared|relative)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Explicit clinical conclusions
    re.compile(
        r"\b(?:clinically significant|not clinically significant|clinical "
        r"relevance|clinical implication|clinical significance)\b",
        re.IGNORECASE,
    ),
    # Study design reference
    re.compile(
        r"\b(?:crossover|parallel|single.dose|multiple.dose|steady.state|"
        r"population PK|PK/PD|PBPK)\b",
        re.IGNORECASE,
    ),
    # Regulatory framing
    re.compile(
        r"\b(?:labeling|dose adjustment|recommend|contraindicate|warn|precaution|"
        r"monitor|avoid|caution)\b",
        re.IGNORECASE,
    ),
    # Mechanistic interpretation
    re.compile(
        r"\b(?:mediated by|inhibit|induce|substrate|pathway|CYP\d[A-Z]\d?|"
        r"P-gp|BCRP|transporter)\b",
        re.IGNORECASE,
    ),
    # Confidence and limitation language
    re.compile(
        r"\b(?:limitation|caveat|confound|generalizab|extrapolat|"
        r"small sample|underpowered|exploratory)\b",
        re.IGNORECASE,
    ),
]

# Patterns that indicate shallow summary rather than analysis
_DEPTH_NEGATIVE_PATTERNS = [
    re.compile(
        r"^(?:the study|the document|the review|this section)\s+(?:shows|states|"
        r"reports|describes|presents|discusses)",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:as stated|as mentioned|as described|according to)\b", re.IGNORECASE),
]

# Minimum response length for a substantive answer (approximate word count)
_MINIMUM_SUBSTANTIVE_WORD_COUNT = 40


# ---------------------------------------------------------------------------
# Coverage scoring — regulatory elements checklist
# ---------------------------------------------------------------------------

# Each entry: (label, pattern, section_types_where_required)
# section_types_where_required=None means checked for all sections
_COVERAGE_ELEMENTS = [
    (
        "labeling_implication",
        re.compile(r"\b(?:label|prescribing information|PI|package insert)\b", re.IGNORECASE),
        ["labeling_recommendations", "drug_drug_interactions", "special_populations"],
    ),
    (
        "dose_recommendation",
        re.compile(
            r"\b(?:dose adjustment|dosing recommendation|modify dose|no dose adjustment)\b",
            re.IGNORECASE,
        ),
        ["labeling_recommendations", "special_populations", "drug_drug_interactions"],
    ),
    (
        "numerical_pk_values",
        re.compile(
            r"\b(?:AUC|Cmax|t1/2|clearance|Vd|bioavailability)\s*(?:of|was|were|increased|decreased)?\s*"
            r"\d+(?:\.\d+)?",
            re.IGNORECASE,
        ),
        ["pk_characteristics", "dose_exposure_response", "drug_drug_interactions"],
    ),
    (
        "clinical_significance_statement",
        re.compile(
            r"\b(?:clinically significant|not clinically significant|clinical relevance)\b",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "study_limitation",
        re.compile(
            r"\b(?:limitation|caveat|small sample|single.dose|exploratory|not powered)\b",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        "population_specificity",
        re.compile(
            r"\b(?:renal|hepatic|pediatric|geriatric|elderly|pregnancy|lactation|"
            r"race|sex|weight|BMI|special population)\b",
            re.IGNORECASE,
        ),
        ["special_populations"],
    ),
]


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------


@dataclass
class DepthEvaluation:
    """Depth scoring result for a single response."""

    raw_score: float
    positive_signals_found: List[str] = field(default_factory=list)
    negative_signals_found: List[str] = field(default_factory=list)
    word_count: int = 0
    is_too_short: bool = False


@dataclass
class CoverageEvaluation:
    """Coverage scoring result for a single response."""

    raw_score: float
    elements_present: List[str] = field(default_factory=list)
    elements_missing: List[str] = field(default_factory=list)
    elements_checked: int = 0


@dataclass
class EvaluationResult:
    """
    Complete quality evaluation for one LLM-generated QnA or comment.
    Returned to the ingestion orchestrator for auto-accept/reject routing.
    """

    content_type: str               # 'qna' or 'comment'
    content_key: str                # qna_key or comment_key

    depth_evaluation: DepthEvaluation
    coverage_evaluation: CoverageEvaluation

    combined_quality_score: float   # 0.0–1.0 combined score
    decision: str                   # DECISION_AUTO_ACCEPT / _REVIEW_PENDING / _AUTO_REJECT
    assigned_source_type: str       # SOURCE_TYPE_* constant
    assigned_benchmark_weight: float

    # Issue flags surfaced in the Review UI
    issue_flags: List[str] = field(default_factory=list)

    # Threshold applied to reach this decision
    threshold_applied: float = 0.85
    auto_reject_threshold: float = 0.50

    section_type: str = ""


# ---------------------------------------------------------------------------
# Depth scoring
# ---------------------------------------------------------------------------


def _evaluate_depth(response_text: str) -> DepthEvaluation:
    """
    Score the analytical depth of a single response text.

    Positive signals add to the score. Negative signals reduce it.
    Very short responses receive a penalty below the auto-reject floor.
    """
    words = response_text.split()
    word_count = len(words)
    is_too_short = word_count < _MINIMUM_SUBSTANTIVE_WORD_COUNT

    if is_too_short:
        return DepthEvaluation(
            raw_score=0.15,
            word_count=word_count,
            is_too_short=True,
        )

    positive_signals: List[str] = []
    negative_signals: List[str] = []

    for pattern in _DEPTH_POSITIVE_PATTERNS:
        match = pattern.search(response_text)
        if match:
            # Capture a short excerpt of the matching text for diagnostics
            excerpt = response_text[max(0, match.start() - 10): match.end() + 20].strip()
            positive_signals.append(excerpt[:80])

    for pattern in _DEPTH_NEGATIVE_PATTERNS:
        match = pattern.search(response_text)
        if match:
            excerpt = response_text[match.start(): match.start() + 60].strip()
            negative_signals.append(excerpt)

    # Base score from positive signal density
    positive_count = len(positive_signals)
    negative_count = len(negative_signals)

    # Normalize: 0 positive = 0.30 floor; 5+ positive = 0.95 ceiling
    positive_contribution = min(0.65, positive_count * 0.13)
    negative_penalty = min(0.30, negative_count * 0.10)

    raw_score = max(0.10, min(1.0, 0.30 + positive_contribution - negative_penalty))

    return DepthEvaluation(
        raw_score=raw_score,
        positive_signals_found=positive_signals[:5],
        negative_signals_found=negative_signals[:3],
        word_count=word_count,
        is_too_short=False,
    )


# ---------------------------------------------------------------------------
# Coverage scoring
# ---------------------------------------------------------------------------


def _evaluate_coverage(
    response_text: str,
    section_type: str,
) -> CoverageEvaluation:
    """
    Score how many required regulatory coverage elements are present.

    Elements are filtered by section_type — only elements required for
    this section type (or required universally) are checked.
    """
    applicable_elements = [
        (label, pattern)
        for label, pattern, required_sections in _COVERAGE_ELEMENTS
        if required_sections is None or section_type in required_sections
    ]

    if not applicable_elements:
        # No required elements for this section — neutral score
        return CoverageEvaluation(
            raw_score=0.75,
            elements_checked=0,
        )

    elements_present: List[str] = []
    elements_missing: List[str] = []

    for label, pattern in applicable_elements:
        if pattern.search(response_text):
            elements_present.append(label)
        else:
            elements_missing.append(label)

    total = len(applicable_elements)
    present_count = len(elements_present)
    raw_score = present_count / total if total > 0 else 0.75

    return CoverageEvaluation(
        raw_score=raw_score,
        elements_present=elements_present,
        elements_missing=elements_missing,
        elements_checked=total,
    )


# ---------------------------------------------------------------------------
# Source type and benchmark weight assignment
# ---------------------------------------------------------------------------


def _assign_source_type_and_weight(
    decision: str,
    settings: Settings,
) -> Tuple:
    """
    Map an evaluation decision to a source_type and benchmark_weight.
    Returns (source_type, benchmark_weight).
    """
    from typing import Tuple

    if decision == DECISION_AUTO_ACCEPT:
        return (
            SOURCE_TYPE_AUTO_REVIEWED,
            settings.benchmark.weight_auto_reviewed,
        )
    else:
        # REVIEW_PENDING and AUTO_REJECT both get auto_generated weight
        # (0.0) until a curator acts on them.
        return (
            SOURCE_TYPE_AUTO_GENERATED,
            settings.benchmark.weight_auto_generated,
        )


# ---------------------------------------------------------------------------
# Issue flag builder
# ---------------------------------------------------------------------------


def _build_issue_flags(
    depth_eval: DepthEvaluation,
    coverage_eval: CoverageEvaluation,
    decision: str,
) -> List[str]:
    """
    Build a list of human-readable issue flags for the Review UI.
    Flags appear next to each QnA/comment in the review panel.
    """
    flags: List[str] = []

    if depth_eval.is_too_short:
        flags.append("Response too short — likely incomplete generation.")

    if depth_eval.negative_signals_found:
        flags.append(
            "Response may be summary rather than analysis — check for interpretive depth."
        )

    if len(depth_eval.positive_signals_found) == 0 and not depth_eval.is_too_short:
        flags.append(
            "No analytical depth signals detected — missing numerical values, "
            "clinical conclusions, or mechanistic explanation."
        )

    for missing_element in coverage_eval.elements_missing:
        readable_label = missing_element.replace("_", " ").capitalize()
        flags.append(f"Missing coverage element: {readable_label}")

    if decision == DECISION_AUTO_REJECT:
        flags.append(
            "AUTO-REJECT: Quality score below minimum threshold. "
            "Requires complete rewrite before commitment."
        )

    return flags


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------


def evaluate_qna(
    question: str,
    answer: str,
    qna_key: str,
    section_type: str,
    settings: Settings,
    benchmark_context: Optional[BenchmarkContext] = None,
    adjusted_accept_threshold: Optional[float] = None,
) -> EvaluationResult:
    """
    Evaluate a single LLM-generated QnA pair for depth and coverage quality.

    Parameters
    ----------
    question : str
        The generated question.
    answer : str
        The generated answer to evaluate.
    qna_key : str
        Key identifier (e.g. 'qna1') for logging and UI display.
    section_type : str
        Normalized section type for coverage element filtering.
    settings : Settings
        Provides quality thresholds and benchmark weights.
    benchmark_context : BenchmarkContext, optional
        Pre-fetched benchmark context. Used for comparison depth checks.
    adjusted_accept_threshold : float, optional
        Coverage-adjusted threshold from benchmark_coverage_map.
        If None, uses settings.ingestion.quality_score_auto_accept.

    Returns
    -------
    EvaluationResult
        Full evaluation with decision and issue flags.
    """
    accept_threshold = (
        adjusted_accept_threshold
        if adjusted_accept_threshold is not None
        else settings.ingestion.quality_score_auto_accept
    )
    reject_threshold = settings.ingestion.quality_score_auto_reject

    depth_eval = _evaluate_depth(answer)
    coverage_eval = _evaluate_coverage(answer, section_type)

    # Combined score: depth weighted slightly higher than coverage for QnA
    combined_score = (depth_eval.raw_score * 0.55) + (coverage_eval.raw_score * 0.45)

    if combined_score >= accept_threshold:
        decision = DECISION_AUTO_ACCEPT
    elif combined_score <= reject_threshold:
        decision = DECISION_AUTO_REJECT
    else:
        decision = DECISION_REVIEW_PENDING

    source_type, benchmark_weight = _assign_source_type_and_weight(decision, settings)
    issue_flags = _build_issue_flags(depth_eval, coverage_eval, decision)

    logger.debug(
        "QnA evaluation: key=%s section=%s score=%.3f decision=%s",
        qna_key,
        section_type,
        combined_score,
        decision,
    )

    return EvaluationResult(
        content_type="qna",
        content_key=qna_key,
        depth_evaluation=depth_eval,
        coverage_evaluation=coverage_eval,
        combined_quality_score=combined_score,
        decision=decision,
        assigned_source_type=source_type,
        assigned_benchmark_weight=benchmark_weight,
        issue_flags=issue_flags,
        threshold_applied=accept_threshold,
        auto_reject_threshold=reject_threshold,
        section_type=section_type,
    )


def evaluate_comment(
    header: str,
    comment: str,
    comment_key: str,
    section_type: str,
    settings: Settings,
    benchmark_context: Optional[BenchmarkContext] = None,
    adjusted_accept_threshold: Optional[float] = None,
) -> EvaluationResult:
    """
    Evaluate a single LLM-generated comment for depth and coverage quality.

    For comments, the combined text (header + comment) is scored together
    since the header provides essential context for the comment's content.

    Parameters
    ----------
    header : str
        The generated comment header/title.
    comment : str
        The generated comment body to evaluate.
    comment_key : str
        Key identifier (e.g. 'comment1') for logging and UI display.
    section_type : str
        Normalized section type for coverage element filtering.
    settings : Settings
        Provides quality thresholds and benchmark weights.
    benchmark_context : BenchmarkContext, optional
        Pre-fetched benchmark context.
    adjusted_accept_threshold : float, optional
        Coverage-adjusted threshold. Falls back to settings value if None.

    Returns
    -------
    EvaluationResult
        Full evaluation with decision and issue flags.
    """
    accept_threshold = (
        adjusted_accept_threshold
        if adjusted_accept_threshold is not None
        else settings.ingestion.quality_score_auto_accept
    )
    reject_threshold = settings.ingestion.quality_score_auto_reject

    combined_text = f"{header}\n{comment}"

    depth_eval = _evaluate_depth(combined_text)
    coverage_eval = _evaluate_coverage(combined_text, section_type)

    # Comments are evaluated with slightly more weight on coverage —
    # their purpose is to flag regulatory elements for the reviewer.
    combined_score = (depth_eval.raw_score * 0.45) + (coverage_eval.raw_score * 0.55)

    if combined_score >= accept_threshold:
        decision = DECISION_AUTO_ACCEPT
    elif combined_score <= reject_threshold:
        decision = DECISION_AUTO_REJECT
    else:
        decision = DECISION_REVIEW_PENDING

    source_type, benchmark_weight = _assign_source_type_and_weight(decision, settings)
    issue_flags = _build_issue_flags(depth_eval, coverage_eval, decision)

    logger.debug(
        "Comment evaluation: key=%s section=%s score=%.3f decision=%s",
        comment_key,
        section_type,
        combined_score,
        decision,
    )

    return EvaluationResult(
        content_type="comment",
        content_key=comment_key,
        depth_evaluation=depth_eval,
        coverage_evaluation=coverage_eval,
        combined_quality_score=combined_score,
        decision=decision,
        assigned_source_type=source_type,
        assigned_benchmark_weight=benchmark_weight,
        issue_flags=issue_flags,
        threshold_applied=accept_threshold,
        auto_reject_threshold=reject_threshold,
        section_type=section_type,
    )


def evaluate_batch(
    items: List[dict],
    content_type: str,
    section_type: str,
    settings: Settings,
    benchmark_context: Optional[BenchmarkContext] = None,
    adjusted_accept_threshold: Optional[float] = None,
) -> List[EvaluationResult]:
    """
    Evaluate a batch of QnA pairs or comments.

    Parameters
    ----------
    items : List[dict]
        For 'qna': each dict has keys 'question', 'answer', 'qna_key'.
        For 'comment': each dict has keys 'header', 'comment', 'comment_key'.
    content_type : str
        'qna' or 'comment'.
    section_type : str
        Applied to all items in the batch.
    settings : Settings
        Shared settings object.
    benchmark_context : BenchmarkContext, optional
        Shared context for all items.
    adjusted_accept_threshold : float, optional
        Shared threshold for all items.

    Returns
    -------
    List[EvaluationResult]
        Same length as items.
    """
    results: List[EvaluationResult] = []

    for item in items:
        if content_type == "qna":
            result = evaluate_qna(
                question=item["question"],
                answer=item["answer"],
                qna_key=item["qna_key"],
                section_type=section_type,
                settings=settings,
                benchmark_context=benchmark_context,
                adjusted_accept_threshold=adjusted_accept_threshold,
            )
        elif content_type == "comment":
            result = evaluate_comment(
                header=item["header"],
                comment=item["comment"],
                comment_key=item["comment_key"],
                section_type=section_type,
                settings=settings,
                benchmark_context=benchmark_context,
                adjusted_accept_threshold=adjusted_accept_threshold,
            )
        else:
            raise ValueError(f"content_type must be 'qna' or 'comment', got: {content_type!r}")

        results.append(result)

    accept_count = sum(1 for r in results if r.decision == DECISION_AUTO_ACCEPT)
    review_count = sum(1 for r in results if r.decision == DECISION_REVIEW_PENDING)
    reject_count = sum(1 for r in results if r.decision == DECISION_AUTO_REJECT)

    logger.info(
        "Batch evaluation complete: type=%s section=%s total=%s "
        "accepted=%s review=%s rejected=%s",
        content_type,
        section_type,
        len(results),
        accept_count,
        review_count,
        reject_count,
    )

    return results
