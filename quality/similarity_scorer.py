"""
quality/similarity_scorer.py

Three-layer composite similarity scoring for the FDA Clinical Pharmacology Pipeline.

Layer 1 — Holistic query similarity            weight: 0.40
  Full query embedding vs response embedding.
  Cosine similarity via sentence-transformers.

Layer 2 — FDA-specific keyword match           weight: 0.35
  spaCy keyword extraction from query.
  Elevated weight for: drug names, formulations, NDA numbers,
  indications, study IDs.
  RapidFuzz fuzzy matching for drug name variants.

Layer 3 — Benchmark alignment                  weight: 0.25
  Retrieve benchmark QnA answers for matched drug and question type.
  Compare response against benchmark answer depth and content coverage.

Composite score evaluated against pipeline.similarity_threshold.
Fail → targeted re-synthesis note identifying which layer failed.
Max 2 retry attempts then fallback response.

This module generates no LLM calls. It uses embedding similarity and
text-match heuristics exclusively. Benchmark alignment calls the
benchmark_retriever directly.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import spacy
from rapidfuzz import fuzz

from config.settings import Settings
from ingestion.benchmark_retriever import (
    BenchmarkContext,
    BenchmarkQnAEntry,
    retrieve_benchmark_qna_entries,
)
from utils.logger import get_module_logger
from vector_store.index_manager import generate_embedding

logger = get_module_logger(__name__)


# ---------------------------------------------------------------------------
# Layer weights — must sum to 1.0
# ---------------------------------------------------------------------------

LAYER_1_HOLISTIC_WEIGHT = 0.40
LAYER_2_KEYWORD_WEIGHT = 0.35
LAYER_3_BENCHMARK_WEIGHT = 0.25

# Fuzzy match threshold for drug name variant matching (RapidFuzz token_sort_ratio)
DRUG_NAME_FUZZY_THRESHOLD = 78

# Elevated keyword categories — these carry more weight in Layer 2 matching
_FDA_HIGH_PRIORITY_ENTITY_LABELS = frozenset([
    "DRUG_NAME", "NDA_BLA", "FORMULATION", "INDICATION", "STUDY_ID",
])

# spaCy model — loaded once at module level
_spacy_nlp = None


def _get_spacy_nlp():
    """Return the loaded spaCy model, loading it on first access."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.debug("spaCy model en_core_web_sm loaded.")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _spacy_nlp


# ---------------------------------------------------------------------------
# FDA-specific term patterns for Layer 2
# ---------------------------------------------------------------------------

_NDA_BLA_PATTERN = re.compile(r"\b(?:NDA|BLA)\s*\d{5,6}\b", re.IGNORECASE)
_STUDY_ID_PATTERN = re.compile(
    r"\b(?:study|trial|protocol)\s+(?:#\s*)?[A-Z0-9\-]{3,20}\b", re.IGNORECASE
)
_DOSE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|mL|g|mg/kg|mg/m2)\b", re.IGNORECASE
)
_AUC_CL_PATTERN = re.compile(
    r"\b(?:AUC|Cmax|Tmax|t1/2|clearance|CL|Vd|bioavailability|half.life)\b",
    re.IGNORECASE,
)

# Regulatory flag terms that should be present in high-quality responses
_REGULATORY_FLAG_TERMS = [
    "labeling", "dose adjustment", "contraindication", "warning",
    "precaution", "special population", "renal impairment", "hepatic impairment",
    "drug interaction", "DDI", "clinical significance", "recommendation",
]


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerScore:
    """Score and diagnostics for a single similarity layer."""

    layer_number: int
    layer_name: str
    raw_score: float            # 0.0–1.0
    weighted_score: float       # raw_score * layer weight
    weight_applied: float
    diagnostic_notes: List[str] = field(default_factory=list)


@dataclass
class SimilarityResult:
    """
    Full composite scoring result for one query–response pair.
    Used by the ingestion pipeline to decide auto-accept, review, or re-synthesis.
    """

    query_text: str
    response_text: str
    composite_score: float          # Weighted sum of all three layer scores
    layer_1_score: LayerScore
    layer_2_score: LayerScore
    layer_3_score: LayerScore
    passes_threshold: bool
    threshold_applied: float
    failed_layers: List[int]        # Layer numbers that contributed to failure
    resynthesis_note: str           # Targeted feedback for re-synthesis prompt
    section_type: str
    drug_name_generic: Optional[str]


# ---------------------------------------------------------------------------
# Layer 1: Holistic embedding similarity
# ---------------------------------------------------------------------------


def _score_layer_1_holistic(
    query_text: str,
    response_text: str,
) -> LayerScore:
    """
    Compute cosine similarity between the query embedding and response embedding.
    Returns a LayerScore with the raw and weighted scores.
    """
    import numpy as np

    query_embedding = generate_embedding(query_text)
    response_embedding = generate_embedding(response_text)

    query_array = np.array(query_embedding)
    response_array = np.array(response_embedding)

    # Embeddings are L2-normalized by generate_embedding(), so dot product = cosine similarity
    cosine_similarity = float(np.dot(query_array, response_array))

    # Clamp to [0.0, 1.0] — normalized embeddings produce values in [-1, 1]
    clamped_score = max(0.0, min(1.0, cosine_similarity))

    notes = [f"Cosine similarity between query and response: {clamped_score:.4f}"]

    return LayerScore(
        layer_number=1,
        layer_name="holistic_embedding_similarity",
        raw_score=clamped_score,
        weighted_score=clamped_score * LAYER_1_HOLISTIC_WEIGHT,
        weight_applied=LAYER_1_HOLISTIC_WEIGHT,
        diagnostic_notes=notes,
    )


# ---------------------------------------------------------------------------
# Layer 2: FDA-specific keyword match
# ---------------------------------------------------------------------------


def _extract_fda_keywords(text: str) -> dict:
    """
    Extract FDA-specific terms from text using spaCy NER and regex patterns.

    Returns a dict with keys:
      nda_bla_numbers: List[str]
      study_ids: List[str]
      dose_values: List[str]
      pk_terms: List[str]
      named_entities: List[str]   (CHEMICAL, ORG, PRODUCT from spaCy)
    """
    nlp = _get_spacy_nlp()
    doc = nlp(text[:5000])  # Limit to 5000 chars for performance

    named_entities = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in ("CHEMICAL", "ORG", "PRODUCT", "GPE")
    ]

    nda_bla_numbers = _NDA_BLA_PATTERN.findall(text)
    study_ids = _STUDY_ID_PATTERN.findall(text)
    dose_values = _DOSE_PATTERN.findall(text)
    pk_terms = _AUC_CL_PATTERN.findall(text)

    return {
        "nda_bla_numbers": nda_bla_numbers,
        "study_ids": study_ids,
        "dose_values": dose_values,
        "pk_terms": pk_terms,
        "named_entities": named_entities,
    }


def _score_layer_2_keyword_match(
    query_text: str,
    response_text: str,
    drug_name_generic: Optional[str],
) -> LayerScore:
    """
    Score keyword coverage: how many of the query's key FDA terms appear
    in the response, with elevated weight for high-priority term types.

    Drug name fuzzy matching via RapidFuzz handles variant forms
    (e.g. "midazolam HCl" vs "midazolam hydrochloride").
    """
    query_keywords = _extract_fda_keywords(query_text)
    response_lower = response_text.lower()

    total_weighted_possible = 0.0
    total_weighted_matched = 0.0
    notes = []

    # --- NDA/BLA numbers (high priority) ---
    nda_weight = 2.0
    for nda_number in query_keywords["nda_bla_numbers"]:
        total_weighted_possible += nda_weight
        if nda_number.lower() in response_lower:
            total_weighted_matched += nda_weight
        else:
            notes.append(f"NDA/BLA number not found in response: {nda_number}")

    # --- Study IDs (high priority) ---
    study_weight = 1.8
    for study_id in query_keywords["study_ids"]:
        total_weighted_possible += study_weight
        if any(word.lower() in response_lower for word in study_id.split()):
            total_weighted_matched += study_weight
        else:
            notes.append(f"Study ID not addressed in response: {study_id}")

    # --- PK parameter terms (medium priority) ---
    pk_weight = 1.5
    for pk_term in set(query_keywords["pk_terms"]):
        total_weighted_possible += pk_weight
        if pk_term.lower() in response_lower:
            total_weighted_matched += pk_weight

    # --- Dose values (medium priority) ---
    dose_weight = 1.2
    for dose_value in set(query_keywords["dose_values"]):
        total_weighted_possible += dose_weight
        if dose_value.lower() in response_lower:
            total_weighted_matched += dose_weight

    # --- Named entities (standard weight) ---
    entity_weight = 1.0
    for entity in set(query_keywords["named_entities"]):
        total_weighted_possible += entity_weight
        entity_lower = entity.lower()
        if entity_lower in response_lower:
            total_weighted_matched += entity_weight
        elif drug_name_generic and fuzz.token_sort_ratio(
            entity_lower, drug_name_generic.lower()
        ) >= DRUG_NAME_FUZZY_THRESHOLD:
            # Fuzzy match for drug name variants
            total_weighted_matched += entity_weight
            notes.append(
                f"Drug name fuzzy matched: {entity!r} ~ {drug_name_generic!r}"
            )

    # --- Drug name direct check (extra coverage) ---
    if drug_name_generic:
        total_weighted_possible += 2.0
        if drug_name_generic.lower() in response_lower:
            total_weighted_matched += 2.0
        else:
            notes.append(
                f"Generic drug name not found in response: {drug_name_generic!r}"
            )

    # Compute final ratio
    if total_weighted_possible == 0.0:
        # No extractable keywords — neutral score, not a failure
        raw_score = 0.70
        notes.append("No extractable FDA keywords found in query — neutral score applied.")
    else:
        raw_score = min(1.0, total_weighted_matched / total_weighted_possible)

    notes.insert(0, f"Keyword match score: {raw_score:.4f}")

    return LayerScore(
        layer_number=2,
        layer_name="fda_keyword_match",
        raw_score=raw_score,
        weighted_score=raw_score * LAYER_2_KEYWORD_WEIGHT,
        weight_applied=LAYER_2_KEYWORD_WEIGHT,
        diagnostic_notes=notes,
    )


# ---------------------------------------------------------------------------
# Layer 3: Benchmark alignment
# ---------------------------------------------------------------------------


def _score_response_against_benchmark_answer(
    response_text: str,
    benchmark_entry: BenchmarkQnAEntry,
) -> float:
    """
    Score how well a response aligns with a single benchmark answer.
    Uses embedding cosine similarity — same as Layer 1 but focused on
    answer-to-answer comparison rather than query-to-response.
    Returns a float in [0.0, 1.0].
    """
    import numpy as np

    benchmark_embedding = generate_embedding(benchmark_entry.answer)
    response_embedding = generate_embedding(response_text)

    benchmark_array = np.array(benchmark_embedding)
    response_array = np.array(response_embedding)

    cosine_similarity = float(np.dot(benchmark_array, response_array))
    return max(0.0, min(1.0, cosine_similarity))


def _score_layer_3_benchmark_alignment(
    query_text: str,
    response_text: str,
    section_type: str,
    drug_name_generic: Optional[str],
    benchmark_context: Optional[BenchmarkContext],
) -> LayerScore:
    """
    Compare the response against the benchmark pool for this section and drug.
    Uses the highest-scoring benchmark match as the alignment score.

    If no benchmark entries are available, applies a neutral score (0.70)
    to avoid penalizing new drugs or sections not yet in the pool.
    """
    notes = []

    # Use provided context or fetch fresh
    if benchmark_context is not None:
        anchors = benchmark_context.qna_entries
    else:
        anchors = retrieve_benchmark_qna_entries(
            section_type=section_type,
            max_entries=3,
            drug_name_generic=drug_name_generic,
        )

    if not anchors:
        raw_score = 0.70
        notes.append(
            "No benchmark entries available for this section — neutral score applied."
        )
        return LayerScore(
            layer_number=3,
            layer_name="benchmark_alignment",
            raw_score=raw_score,
            weighted_score=raw_score * LAYER_3_BENCHMARK_WEIGHT,
            weight_applied=LAYER_3_BENCHMARK_WEIGHT,
            diagnostic_notes=notes,
        )

    alignment_scores = []
    for anchor in anchors:
        individual_score = _score_response_against_benchmark_answer(
            response_text=response_text,
            benchmark_entry=anchor,
        )
        alignment_scores.append(individual_score)
        notes.append(
            f"Benchmark qna_id={anchor.qna_id} "
            f"(weight={anchor.benchmark_weight:.2f}): "
            f"alignment={individual_score:.4f}"
        )

    # Weight each benchmark score by its benchmark_weight before taking max
    weighted_alignment_scores = [
        score * anchor.benchmark_weight
        for score, anchor in zip(alignment_scores, anchors)
    ]
    raw_score = min(1.0, max(weighted_alignment_scores)) if weighted_alignment_scores else 0.70

    notes.insert(0, f"Benchmark alignment score: {raw_score:.4f}")

    return LayerScore(
        layer_number=3,
        layer_name="benchmark_alignment",
        raw_score=raw_score,
        weighted_score=raw_score * LAYER_3_BENCHMARK_WEIGHT,
        weight_applied=LAYER_3_BENCHMARK_WEIGHT,
        diagnostic_notes=notes,
    )


# ---------------------------------------------------------------------------
# Re-synthesis note builder
# ---------------------------------------------------------------------------


def _build_resynthesis_note(
    layer_1: LayerScore,
    layer_2: LayerScore,
    layer_3: LayerScore,
    threshold: float,
    composite_score: float,
) -> str:
    """
    Build a targeted re-synthesis note that names which layer(s) failed
    and what specific aspects need improvement.

    This note is injected into the re-synthesis prompt so the LLM knows
    exactly what to fix.
    """
    lines = [
        f"Quality score {composite_score:.3f} did not meet threshold {threshold:.3f}.",
        "Specific issues to address in your revised response:",
        "",
    ]

    if layer_1.raw_score < 0.60:
        lines.append(
            "- RELEVANCE: Your response does not sufficiently address the question asked. "
            "Re-read the question carefully and ensure the answer is directly on-topic."
        )

    if layer_2.raw_score < 0.60:
        failed_keyword_notes = [
            note for note in layer_2.diagnostic_notes
            if "not found" in note or "not addressed" in note
        ]
        if failed_keyword_notes:
            lines.append(
                "- CONTENT GAPS: The following FDA-specific elements from the question "
                "were not addressed in your response:"
            )
            for note in failed_keyword_notes[:5]:
                lines.append(f"    {note}")

    if layer_3.raw_score < 0.60:
        lines.append(
            "- DEPTH: Your response lacks the analytical depth expected for this "
            "section type. Include: specific study findings with numerical values, "
            "regulatory significance of the findings, clinical implications, "
            "and any study limitations relevant to the conclusions."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def score_query_response(
    query_text: str,
    response_text: str,
    section_type: str,
    settings: Settings,
    drug_name_generic: Optional[str] = None,
    benchmark_context: Optional[BenchmarkContext] = None,
) -> SimilarityResult:
    """
    Compute the three-layer composite similarity score for a query–response pair.

    Parameters
    ----------
    query_text : str
        The question or query submitted to the LLM.
    response_text : str
        The LLM-generated response to evaluate.
    section_type : str
        Normalized section type (e.g. 'pk_characteristics').
    settings : Settings
        Provides pipeline.similarity_threshold.
    drug_name_generic : str, optional
        Used for keyword matching and benchmark retrieval.
    benchmark_context : BenchmarkContext, optional
        Pre-fetched benchmark context. If None, Layer 3 fetches its own.

    Returns
    -------
    SimilarityResult
        Full scoring breakdown with pass/fail decision and re-synthesis note.
    """
    threshold = settings.pipeline.similarity_threshold

    layer_1 = _score_layer_1_holistic(query_text, response_text)
    layer_2 = _score_layer_2_keyword_match(query_text, response_text, drug_name_generic)
    layer_3 = _score_layer_3_benchmark_alignment(
        query_text, response_text, section_type, drug_name_generic, benchmark_context
    )

    composite_score = (
        layer_1.weighted_score
        + layer_2.weighted_score
        + layer_3.weighted_score
    )

    passes_threshold = composite_score >= threshold

    failed_layers = []
    if layer_1.raw_score < 0.60:
        failed_layers.append(1)
    if layer_2.raw_score < 0.60:
        failed_layers.append(2)
    if layer_3.raw_score < 0.60:
        failed_layers.append(3)

    resynthesis_note = ""
    if not passes_threshold:
        resynthesis_note = _build_resynthesis_note(
            layer_1, layer_2, layer_3, threshold, composite_score
        )

    logger.info(
        "Similarity score: composite=%.3f threshold=%.3f pass=%s "
        "section=%s drug=%s layers=[%.3f, %.3f, %.3f]",
        composite_score,
        threshold,
        passes_threshold,
        section_type,
        drug_name_generic,
        layer_1.raw_score,
        layer_2.raw_score,
        layer_3.raw_score,
    )

    return SimilarityResult(
        query_text=query_text,
        response_text=response_text,
        composite_score=composite_score,
        layer_1_score=layer_1,
        layer_2_score=layer_2,
        layer_3_score=layer_3,
        passes_threshold=passes_threshold,
        threshold_applied=threshold,
        failed_layers=failed_layers,
        resynthesis_note=resynthesis_note,
        section_type=section_type,
        drug_name_generic=drug_name_generic,
    )
