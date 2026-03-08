## Ingestion Pipeline Flow
```
PDF dropped into ingestion folder
            │
            ▼
┌─────────────────────────────────────────────┐
│  Job Registration                           │
│  Write ingestion_jobs record                │
│  status: 'started'                          │
│  Check for existing resumable job first     │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  Step 1 — Raw Text Extraction               │
│  PyMuPDF / pdfplumber                       │
│  Update job: status 'extracting'            │
│  Log: pages extracted, structure detected   │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  Step 2 — Three-Level Section Chunking      │
│  Build Level 1, 2, 3 chunk hierarchy        │
│  Tag each chunk with parent metadata        │
│  Update job: status 'chunking'              │
└──────────────────────┬──────────────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
┌───────────────────┐   ┌─────────────────────────────────┐
│  Step 3A          │   │  Step 3B                        │
│  Metadata         │   │  Benchmark Retrieval            │
│  Extraction       │   │  Query coverage_map             │
│  Regex + rules    │   │  Adjust thresholds by coverage  │
│  → pdf_metadata   │   │  Retrieve 3-5 anchors per       │
│  Pydantic valid.  │   │  section type                   │
│  Update job:      │   │  Update job:                    │
│  'metadata_       │   │  'benchmark_retrieved'          │
│  extracted'       │   │                                 │
└─────────┬─────────┘   └────────────────┬────────────────┘
          │                              │
          └──────────────┬───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────┐
│  Step 4 — Section Routing                   │
│  high_priority   → LLM processing           │
│  rule_based_only → regex extraction only    │
│  skip            → omit from processing     │
│  Update job: 'generating_qna'               │
└──────────────────────┬──────────────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
┌───────────────────────┐  ┌──────────────────────────┐
│  Step 5A              │  │  Step 5B                  │
│  LLM QnA Generation   │  │  LLM Comment Generation   │
│  Level 2 chunks       │  │  Level 2 chunks           │
│  + benchmark anchors  │  │  + benchmark anchors      │
│  + persona prompt     │  │  + persona prompt         │
│  + cross-doc context  │  │  + cross-doc context      │
│  Mistral: factual PK  │  │  GPT-4o / Claude: all     │
│  GPT-4o: DDI, labels  │  │  sections                 │
│  Update job: status   │  │  Update job: status       │
└──────────┬────────────┘  └─────────────┬────────────┘
           │                             │
           └──────────────┬──────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────┐
│  Step 6 — Quality Scoring                   │
│  Depth score + coverage score               │
│  Compare against benchmark entries          │
│  Apply coverage-adjusted thresholds         │
│  Update job: 'quality_scoring'              │
│                                             │
│  Pass threshold  → stage for review         │
│  Fail threshold  → targeted revision prompt │
│                   → retry once              │
│                   → if still fails          │
│                     → flag for mandatory    │
│                       curator review        │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  Step 7 — Streamlit Ingestion Review UI     │
│  Update job: 'review_pending'               │
│  Curator reviews, edits, accepts, deletes   │
│  All edits written to content_versions      │
│  before updating live record                │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  Step 8 — Two-Phase Commit                  │
│                                             │
│  Phase 1: Stage all writes                  │
│  Write pdf_metadata to PostgreSQL           │
│  Write qna_store records to PostgreSQL      │
│  Write comments_store records to PostgreSQL │
│  Write content_versions for all entries     │
│  If Phase 1 fails → rollback, log error     │
│                                             │
│  Phase 2: Vector writes                     │
│  Write all embeddings to ChromaDB           │
│  If Phase 2 fails → rollback Phase 1        │
│                     log error with detail   │
│                     retain job for retry    │
│                                             │
│  Both succeed → finalize                    │
│  Update coverage_map                        │
│  Check knowledge_base_nominations           │
│  Update job: 'committed'                    │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  Step 9 — Post-Commit Actions               │
│  Set decay_review_due on all new entries    │
│  Update benchmark_coverage_map              │
│  Run nomination check:                      │
│    Find entries across docs that agree      │
│    Write to knowledge_base_nominations      │
│  Write final ingestion log entry            │
└─────────────────────────────────────────────┘
```

---

## Stage 3A — Analysis Block at Query Time

Incorporates the three-level chunk retrieval and coverage-aware trust hierarchy:
```
User query (structured intent object)
              │
              ▼
┌──────────────────────────────────────────────┐
│  Level 3 semantic search                     │
│  Embed query → search ChromaDB               │
│  Return top-N paragraph-level chunks         │
│  Each chunk carries Level 2 + Level 1 tags   │
└─────────────────────┬────────────────────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
┌──────────────────┐   ┌────────────────────────┐
│  Direct SQL      │   │  Benchmark retrieval    │
│  query on        │   │  Retrieve curated QnA   │
│  qna_store and   │   │  and comments for       │
│  comments_store  │   │  matched drug/section   │
│  WHERE           │   │  benchmark_eligible     │
│  drug name AND   │   │  = TRUE                 │
│  formulation AND │   │  ORDER BY               │
│  nda_number      │   │  benchmark_weight DESC  │
│  exact match     │   │                         │
└────────┬─────────┘   └───────────┬─────────────┘
         │                         │
         └──────────────┬──────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│  Trust hierarchy merge                       │
│                                              │
│  Priority 1: human_curated QnA answers       │
│  Priority 2: auto_corrected QnA answers      │
│  Priority 3: human_curated comments          │
│  Priority 4: knowledge_base verified entries │
│  Priority 5: Level 3 semantic chunks         │
│  Priority 6: auto_reviewed entries           │
│  Priority 7: online_promoted KB entries      │
│                                              │
│  Pass merged, ranked context to Synthesis    │
└──────────────────────────────────────────────┘
```

---

##  Stage 5 — Similarity Check

Now incorporates benchmark QnA answers as a quality reference alongside the user query matching:
```
Synthesized response
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
┌────────────────────────┐         ┌──────────────────────────────┐
│  Layer 1               │         │  Layer 3 (new)               │
│  Holistic similarity   │         │  Benchmark alignment         │
│  Full query embedding  │         │  Retrieve benchmark QnA      │
│  vs response embedding │         │  answers for this drug and   │
│  Cosine similarity     │         │  question type               │
│  score                 │         │  Compare response against    │
│                        │         │  how benchmark answers       │
│                        │         │  address similar questions   │
│                        │         │  Score: alignment with       │
│                        │         │  benchmark answer style      │
│                        │         │  and content depth           │
└───────────┬────────────┘         └─────────────┬────────────────┘
            │                                     │
            ▼                                     │
┌────────────────────────┐                        │
│  Layer 2               │                        │
│  Keyword match         │                        │
│  spaCy extraction      │                        │
│  FDA-specific signals: │                        │
│  drug name, formulation│                        │
│  NDA number, indication│                        │
│  RapidFuzz matching    │                        │
└───────────┬────────────┘                        │
            │                                     │
            └──────────────┬──────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │  Composite score             │
            │  Layer 1 weight: 0.40        │
            │  Layer 2 weight: 0.35        │
            │  Layer 3 weight: 0.25        │
            └──────────────┬───────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
         Pass threshold           Fail threshold
              │                         │
              ▼                         ▼
     Format and output          Targeted re-synthesis
                                 note: what failed
                                 Layer 1, 2, or 3
                                 Max 2 retries
                                 then fallback response
```
