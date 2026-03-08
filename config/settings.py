"""
config/settings.py

Loads all configuration for the FDA Clinical Pharmacology Pipeline.

Rules enforced here:
- All sensitive values come exclusively from .env via python-dotenv.
- config.yaml holds only non-sensitive tunable parameters.
- No hardcoded secrets or credentials anywhere in this file.
- This module provides a single Settings object imported by all other modules.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv

from utils.logger import get_module_logger

logger = get_module_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# Dataclasses for typed config sections
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    max_resynthesis_attempts: int
    similarity_threshold: float
    benchmark_minimum_coverage: int


@dataclass
class IngestionConfig:
    chunk_size_level2_tokens: int
    chunk_overlap_level2_tokens: int
    chunk_size_level3_tokens: int
    chunk_overlap_level3_tokens: int
    max_qna_pairs_per_section: int
    max_comments_per_section: int
    quality_score_auto_accept: float
    quality_score_auto_reject: float


@dataclass
class LLMConfig:
    mode: str
    local_model: str
    local_endpoint: str
    api_model: str
    api_provider: str


@dataclass
class SectionsConfig:
    high_priority: List[str]
    rule_based_only: List[str]
    skip: List[str]


@dataclass
class BenchmarkConfig:
    decay_review_days: int
    weight_human_curated: float
    weight_auto_corrected: float
    weight_auto_reviewed: float
    weight_auto_generated: float


@dataclass
class ChunkingLevels:
    level_1: str
    level_2: str
    level_3: str


@dataclass
class ChunkingConfig:
    levels: ChunkingLevels


@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def connection_string(self) -> str:
        """
        Construct the SQLAlchemy-compatible PostgreSQL connection string.
        Never logged or stored outside this property.
        """
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class APIKeysConfig:
    openai_api_key: str
    anthropic_api_key: str


@dataclass
class StorageConfig:
    chromadb_path: str
    pdf_storage_path: str


@dataclass
class Settings:
    pipeline: PipelineConfig
    ingestion: IngestionConfig
    llm: LLMConfig
    sections: SectionsConfig
    benchmark: BenchmarkConfig
    chunking: ChunkingConfig
    database: DatabaseConfig
    api_keys: APIKeysConfig
    storage: StorageConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _require_env(variable_name: str) -> str:
    """
    Return the value of an environment variable.
    Raises a descriptive RuntimeError if the variable is absent or empty.
    """
    value = os.getenv(variable_name, "").strip()
    if not value:
        raise RuntimeError(
            f"Required environment variable '{variable_name}' is missing or empty. "
            f"Check your .env file."
        )
    return value


def load_settings() -> Settings:
    """
    Load and return a fully populated Settings object.

    Reads:
      - Sensitive values from .env (via python-dotenv)
      - Non-sensitive tunable parameters from config/config.yaml

    Returns
    -------
    Settings
        Immutable configuration object used throughout the pipeline.
    """
    load_dotenv(dotenv_path=_ENV_PATH)
    logger.info("Environment variables loaded from %s", _ENV_PATH)

    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {_CONFIG_PATH}")

    with open(_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    logger.info("config.yaml loaded from %s", _CONFIG_PATH)

    # --- Non-sensitive config from config.yaml ---

    pipeline_config = PipelineConfig(
        max_resynthesis_attempts=raw_config["pipeline"]["max_resynthesis_attempts"],
        similarity_threshold=raw_config["pipeline"]["similarity_threshold"],
        benchmark_minimum_coverage=raw_config["pipeline"]["benchmark_minimum_coverage"],
    )

    ingestion_config = IngestionConfig(
        chunk_size_level2_tokens=raw_config["ingestion"]["chunk_size_level2_tokens"],
        chunk_overlap_level2_tokens=raw_config["ingestion"]["chunk_overlap_level2_tokens"],
        chunk_size_level3_tokens=raw_config["ingestion"]["chunk_size_level3_tokens"],
        chunk_overlap_level3_tokens=raw_config["ingestion"]["chunk_overlap_level3_tokens"],
        max_qna_pairs_per_section=raw_config["ingestion"]["max_qna_pairs_per_section"],
        max_comments_per_section=raw_config["ingestion"]["max_comments_per_section"],
        quality_score_auto_accept=raw_config["ingestion"]["quality_score_auto_accept"],
        quality_score_auto_reject=raw_config["ingestion"]["quality_score_auto_reject"],
    )

    llm_config = LLMConfig(
        mode=raw_config["llm"]["mode"],
        local_model=raw_config["llm"]["local_model"],
        local_endpoint=raw_config["llm"]["local_endpoint"],
        api_model=raw_config["llm"]["api_model"],
        api_provider=raw_config["llm"]["api_provider"],
    )

    sections_config = SectionsConfig(
        high_priority=raw_config["sections"]["high_priority"],
        rule_based_only=raw_config["sections"]["rule_based_only"],
        skip=raw_config["sections"]["skip"],
    )

    benchmark_config = BenchmarkConfig(
        decay_review_days=raw_config["benchmark"]["decay_review_days"],
        weight_human_curated=raw_config["benchmark"]["weight_human_curated"],
        weight_auto_corrected=raw_config["benchmark"]["weight_auto_corrected"],
        weight_auto_reviewed=raw_config["benchmark"]["weight_auto_reviewed"],
        weight_auto_generated=raw_config["benchmark"]["weight_auto_generated"],
    )

    chunking_config = ChunkingConfig(
        levels=ChunkingLevels(
            level_1=raw_config["chunking"]["levels"]["level_1"],
            level_2=raw_config["chunking"]["levels"]["level_2"],
            level_3=raw_config["chunking"]["levels"]["level_3"],
        )
    )

    # --- Sensitive values from .env only ---

    database_config = DatabaseConfig(
        host=_require_env("POSTGRES_HOST"),
        port=int(_require_env("POSTGRES_PORT")),
        database=_require_env("POSTGRES_DB"),
        user=_require_env("POSTGRES_USER"),
        password=_require_env("POSTGRES_PASSWORD"),
    )

    api_keys_config = APIKeysConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )

    storage_config = StorageConfig(
        chromadb_path=_require_env("CHROMADB_PATH"),
        pdf_storage_path=_require_env("PDF_STORAGE_PATH"),
    )

    settings = Settings(
        pipeline=pipeline_config,
        ingestion=ingestion_config,
        llm=llm_config,
        sections=sections_config,
        benchmark=benchmark_config,
        chunking=chunking_config,
        database=database_config,
        api_keys=api_keys_config,
        storage=storage_config,
    )

    logger.info("Settings loaded successfully")
    return settings
