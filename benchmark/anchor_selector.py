"""
benchmark/anchor_selector.py

Benchmark anchor selection for the FDA Clinical Pharmacology Pipeline.

The anchor selector is responsible for choosing *which* benchmark entries
from the pool to inject as few-shot anchors into LLM prompts. It is distinct
from benchmark_retriever.py, which performs raw database retrieval.

The selector applies two additional criteria on top of retrieval:

  1. Relevance ranking:
     Entries are re-ranked by embedding similarity to the current chunk text,
     not just by benchmark_weight alone. A high-weight entry that is
     thematically distant from the current chunk is a poor few-shot anchor.

  2. Diversity enforcement:
     If multiple high-scoring anchors are near-duplicates (cosine similarity
     above a threshold), the lower-ranked duplicate is dropped.
     This prevents the prompt from being saturated with redundant examples.

The selector returns an ordered AnchorSet: the first entry is the most
relevant anchor, subsequent entries add thematic diversity.

Usage:
  anchor_set = select_anchors_for_chunk(
      chunk_text=chunk.text,
      section_type=chunk.section_type,
      drug_name_generic=drug_name_generic,
      settings=settings,
  )
  if anchor_set.has_anchors():
      prompt_block = anchor_set.format_for_prompt()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from ingestion.benchmark_retriever import (
    BenchmarkCommentEntry,
    BenchmarkQnAEntry,
    retrieve_benchmark_comment_entries,
    retrieve_benchmark_qna_entries,
)
from utils.logger import get_module_logger
from vector_store.index_manager import generate_embedding, generate_embeddings_batch

logger = get_module_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Minimum cosine similarity between chunk and anchor to include the anchor.
# Below this threshold, the anchor is considered thematically irrelevant.
MINIMUM_ANCHOR_RELEVANCE_SCORE = 0.35

# Maximum cosine similarity between two anchors before the lower-ranked
# one is considered a duplicate and dropped.
MAXIMUM_INTER_ANCHOR_SIMILARITY = 0.88

# Pool size multiplier: retrieve this many candidates, then re-rank down to max_anchors.
# Retrieves 3x as many candidates as needed, providing enough options to
# enforce diversity and relevance filtering without running out.
CANDIDATE_POOL_MULTIPLIER = 3


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------


@dataclass
class RankedAnchor:
    """
    A single benchmark entry that has been relevance-ranked and selected.
    Wraps either a BenchmarkQnAEntry or BenchmarkCommentEntry.
    """

    entry_type: str                 # 'qna' or 'comment'
    record_id: int                  # qna_id or comment_id
    relevance_score: float          # Cosine similarity to the current chunk
    benchmark_weight: float         # Original benchmark weight from the pool
    combined_score: float           # Weighted combination for final ranking
    section_type: str
    source_type: str

    # Content fields — populated from the underlying entry
    primary_text: str               # question (QnA) or header (comment)
    secondary_text: str             # answer (QnA) or comment body (comment)


@dataclass
class AnchorSet:
    """
    Ordered set of selected benchmark anchors for a single chunk.
    First entry is most relevant. Subsequent entries add diversity.
    """

    section_type: str
    chunk_text_preview: str         # First 80 chars of the source chunk
    anchors: List[RankedAnchor] = field(default_factory=list)
    candidates_evaluated: int = 0
    candidates_dropped_for_relevance: int = 0
    candidates_dropped_for_duplicates: int = 0

    def has_anchors(self) -> bool:
        return len(self.anchors) > 0

    def format_for_prompt(self) -> str:
        """
        Format the anchor set as a labeled prompt block for LLM injection.
        Returns empty string if no anchors are available.
        """
        if not self.anchors:
            return ""

        lines = [
            "BENCHMARK EXAMPLES — these represent the expected analytical quality. "
            "Match this depth, specificity, and regulatory framing:\n"
        ]

        for index, anchor in enumerate(self.anchors, start=1):
            if anchor.entry_type == "qna":
                lines.append(f"Example {index} (QnA):")
                lines.append(f"  Question: {anchor.primary_text}")
                lines.append(f"  Answer: {anchor.secondary_text}")
            else:
                lines.append(f"Example {index} (Comment):")
                lines.append(f"  Header: {anchor.primary_text}")
                lines.append(f"  Comment: {anchor.secondary_text}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring and ranking helpers
# ---------------------------------------------------------------------------


def _compute_combined_score(
    relevance_score: float,
    benchmark_weight: float,
) -> float:
    """
    Combine relevance score and benchmark weight into a single ranking score.

    The combination weights relevance (semantic match to current chunk)
    more heavily than benchmark weight (source quality) for anchor selection.
    Rationale: a highly relevant lower-quality anchor is more useful for
    few-shot injection than an irrelevant perfect human-curated example.

    relevance_score weight: 0.65
    benchmark_weight weight: 0.35
    """
    return (relevance_score * 0.65) + (benchmark_weight * 0.35)


def _is_duplicate_anchor(
    new_anchor_embedding: List[float],
    existing_anchor_embeddings: List[List[float]],
) -> bool:
    """
    Return True if the new anchor is too similar to any already-selected anchor.
    Uses cosine similarity against all existing anchor embeddings.
    """
    if not existing_anchor_embeddings:
        return False

    new_array = np.array(new_anchor_embedding)
    for existing_embedding in existing_anchor_embeddings:
        existing_array = np.array(existing_embedding)
        similarity = float(np.dot(new_array, existing_array))
        if similarity >= MAXIMUM_INTER_ANCHOR_SIMILARITY:
            return True
    return False


# ---------------------------------------------------------------------------
# QnA anchor selection
# ---------------------------------------------------------------------------


def _select_qna_anchors(
    chunk_text: str,
    chunk_embedding: List[float],
    section_type: str,
    drug_name_generic: Optional[str],
    max_anchors: int,
) -> tuple:
    """
    Retrieve, re-rank, and select QnA anchors for a chunk.
    Returns (List[RankedAnchor], candidates_evaluated, dropped_relevance, dropped_duplicates).
    """
    candidate_count = max_anchors * CANDIDATE_POOL_MULTIPLIER

    candidate_entries: List[BenchmarkQnAEntry] = retrieve_benchmark_qna_entries(
        section_type=section_type,
        max_entries=candidate_count,
        drug_name_generic=drug_name_generic,
    )

    if not candidate_entries:
        return [], 0, 0, 0

    # Batch embed all candidate answers
    answer_texts = [entry.answer for entry in candidate_entries]
    answer_embeddings = generate_embeddings_batch(answer_texts)

    chunk_array = np.array(chunk_embedding)

    ranked_candidates = []
    for entry, answer_embedding in zip(candidate_entries, answer_embeddings):
        answer_array = np.array(answer_embedding)
        relevance_score = max(0.0, float(np.dot(chunk_array, answer_array)))
        combined_score = _compute_combined_score(relevance_score, entry.benchmark_weight)
        ranked_candidates.append((entry, answer_embedding, relevance_score, combined_score))

    # Sort by combined score descending
    ranked_candidates.sort(key=lambda item: item[3], reverse=True)

    selected_anchors: List[RankedAnchor] = []
    selected_embeddings: List[List[float]] = []
    dropped_relevance = 0
    dropped_duplicates = 0

    for entry, answer_embedding, relevance_score, combined_score in ranked_candidates:
        if len(selected_anchors) >= max_anchors:
            break

        if relevance_score < MINIMUM_ANCHOR_RELEVANCE_SCORE:
            dropped_relevance += 1
            continue

        if _is_duplicate_anchor(answer_embedding, selected_embeddings):
            dropped_duplicates += 1
            continue

        selected_anchors.append(
            RankedAnchor(
                entry_type="qna",
                record_id=entry.qna_id,
                relevance_score=relevance_score,
                benchmark_weight=entry.benchmark_weight,
                combined_score=combined_score,
                section_type=entry.section_type,
                source_type=entry.source_type,
                primary_text=entry.question,
                secondary_text=entry.answer,
            )
        )
        selected_embeddings.append(answer_embedding)

    return (
        selected_anchors,
        len(candidate_entries),
        dropped_relevance,
        dropped_duplicates,
    )


# ---------------------------------------------------------------------------
# Comment anchor selection
# ---------------------------------------------------------------------------


def _select_comment_anchors(
    chunk_text: str,
    chunk_embedding: List[float],
    section_type: str,
    drug_name_generic: Optional[str],
    max_anchors: int,
) -> tuple:
    """
    Retrieve, re-rank, and select comment anchors for a chunk.
    Returns (List[RankedAnchor], candidates_evaluated, dropped_relevance, dropped_duplicates).
    """
    candidate_count = max_anchors * CANDIDATE_POOL_MULTIPLIER

    candidate_entries: List[BenchmarkCommentEntry] = retrieve_benchmark_comment_entries(
        section_type=section_type,
        max_entries=candidate_count,
        drug_name_generic=drug_name_generic,
    )

    if not candidate_entries:
        return [], 0, 0, 0

    # Embed combined header + comment text for richer similarity
    combined_texts = [
        f"{entry.header}\n{entry.comment}" for entry in candidate_entries
    ]
    combined_embeddings = generate_embeddings_batch(combined_texts)

    chunk_array = np.array(chunk_embedding)

    ranked_candidates = []
    for entry, combined_embedding in zip(candidate_entries, combined_embeddings):
        combined_array = np.array(combined_embedding)
        relevance_score = max(0.0, float(np.dot(chunk_array, combined_array)))
        combined_score = _compute_combined_score(relevance_score, entry.benchmark_weight)
        ranked_candidates.append((entry, combined_embedding, relevance_score, combined_score))

    ranked_candidates.sort(key=lambda item: item[3], reverse=True)

    selected_anchors: List[RankedAnchor] = []
    selected_embeddings: List[List[float]] = []
    dropped_relevance = 0
    dropped_duplicates = 0

    for entry, combined_embedding, relevance_score, combined_score in ranked_candidates:
        if len(selected_anchors) >= max_anchors:
            break

        if relevance_score < MINIMUM_ANCHOR_RELEVANCE_SCORE:
            dropped_relevance += 1
            continue

        if _is_duplicate_anchor(combined_embedding, selected_embeddings):
            dropped_duplicates += 1
            continue

        selected_anchors.append(
            RankedAnchor(
                entry_type="comment",
                record_id=entry.comment_id,
                relevance_score=relevance_score,
                benchmark_weight=entry.benchmark_weight,
                combined_score=combined_score,
                section_type=entry.section_type,
                source_type=entry.source_type,
                primary_text=entry.header,
                secondary_text=entry.comment,
            )
        )
        selected_embeddings.append(combined_embedding)

    return (
        selected_anchors,
        len(candidate_entries),
        dropped_relevance,
        dropped_duplicates,
    )


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------


def select_anchors_for_chunk(
    chunk_text: str,
    section_type: str,
    settings,
    drug_name_generic: Optional[str] = None,
    max_qna_anchors: int = 2,
    max_comment_anchors: int = 2,
) -> AnchorSet:
    """
    Select the most relevant, diverse benchmark anchors for a document chunk.

    This is the primary entry point called by ingestion stages before
    building LLM prompts. It runs relevance re-ranking and diversity
    filtering on top of the raw pool retrieval from benchmark_retriever.py.

    Parameters
    ----------
    chunk_text : str
        The Level-2 chunk text to be processed by the LLM. Anchors are
        selected for relevance to this specific chunk, not the document overall.
    section_type : str
        Normalized section type for pool filtering.
    settings : Settings
        Provides configuration values.
    drug_name_generic : str, optional
        Used for drug-preference retrieval and logged for diagnostics.
    max_qna_anchors : int
        Maximum QnA anchors to include. Default 2.
    max_comment_anchors : int
        Maximum comment anchors to include. Default 2.

    Returns
    -------
    AnchorSet
        Ordered set of selected anchors. Call has_anchors() before use.
        Call format_for_prompt() to get the injection-ready text block.
    """
    # Embed the chunk once — shared across QnA and comment selection
    chunk_embedding = generate_embedding(chunk_text)

    qna_anchors, qna_candidates, qna_dropped_rel, qna_dropped_dup = (
        _select_qna_anchors(
            chunk_text=chunk_text,
            chunk_embedding=chunk_embedding,
            section_type=section_type,
            drug_name_generic=drug_name_generic,
            max_anchors=max_qna_anchors,
        )
    )

    comment_anchors, comment_candidates, comment_dropped_rel, comment_dropped_dup = (
        _select_comment_anchors(
            chunk_text=chunk_text,
            chunk_embedding=chunk_embedding,
            section_type=section_type,
            drug_name_generic=drug_name_generic,
            max_anchors=max_comment_anchors,
        )
    )

    all_anchors = qna_anchors + comment_anchors

    anchor_set = AnchorSet(
        section_type=section_type,
        chunk_text_preview=chunk_text[:80],
        anchors=all_anchors,
        candidates_evaluated=qna_candidates + comment_candidates,
        candidates_dropped_for_relevance=qna_dropped_rel + comment_dropped_rel,
        candidates_dropped_for_duplicates=qna_dropped_dup + comment_dropped_dup,
    )

    logger.debug(
        "Anchor selection: section=%s drug=%s anchors=%s "
        "candidates=%s dropped_relevance=%s dropped_duplicates=%s",
        section_type,
        drug_name_generic,
        len(all_anchors),
        anchor_set.candidates_evaluated,
        anchor_set.candidates_dropped_for_relevance,
        anchor_set.candidates_dropped_for_duplicates,
    )

    return anchor_set


def select_anchors_for_quality_scoring(
    response_text: str,
    section_type: str,
    drug_name_generic: Optional[str],
    max_anchors: int = 3,
) -> AnchorSet:
    """
    Select benchmark anchors for quality scoring reference (Layer 3 of similarity_scorer).

    Same relevance-ranking logic as select_anchors_for_chunk but
    embeds the *response* text to find anchors most similar to what
    was actually generated — used as the comparison baseline.

    Parameters
    ----------
    response_text : str
        The LLM-generated response to score.
    section_type : str
        Normalized section type.
    drug_name_generic : str, optional
        Drug preference for retrieval.
    max_anchors : int
        Total anchors to return (QnA only — responses are scored against QnA answers).

    Returns
    -------
    AnchorSet
        Anchors for use as the Layer 3 benchmark alignment reference.
    """
    response_embedding = generate_embedding(response_text)

    qna_anchors, qna_candidates, qna_dropped_rel, qna_dropped_dup = (
        _select_qna_anchors(
            chunk_text=response_text,
            chunk_embedding=response_embedding,
            section_type=section_type,
            drug_name_generic=drug_name_generic,
            max_anchors=max_anchors,
        )
    )

    return AnchorSet(
        section_type=section_type,
        chunk_text_preview=response_text[:80],
        anchors=qna_anchors,
        candidates_evaluated=qna_candidates,
        candidates_dropped_for_relevance=qna_dropped_rel,
        candidates_dropped_for_duplicates=qna_dropped_dup,
    )
