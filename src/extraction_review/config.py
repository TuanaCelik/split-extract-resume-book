"""
Configuration for the extraction review application.

Configuration is loaded from configs/config.json via ResourceConfig.
The unified config contains both extraction settings and the JSON schema.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from .json_util import create_union_schema as create_union_schema
from .json_util import get_extraction_schema as get_extraction_schema

logger = logging.getLogger(__name__)


# The name of the collection to use for storing extracted data.
# When developing locally, this will use the _public collection (shared within the project),
# otherwise agent data is isolated to each agent.
EXTRACTED_DATA_COLLECTION: str = "extraction-review"


class ExtractSettings(BaseModel):
    """Extraction settings loaded from configs/config.json extract.settings."""

    extraction_mode: Literal["FAST", "PREMIUM", "MULTIMODAL"]
    system_prompt: str | None = None
    citation_bbox: bool = False
    use_reasoning: bool = False
    cite_sources: bool = False
    confidence_scores: bool = False


class ExtractConfig(BaseModel):
    """Full extraction configuration with schema and settings."""

    json_schema: dict[str, Any]
    settings: ExtractSettings


class SplitCategory(BaseModel):
    """A category for document splitting."""

    name: str
    description: str


class SplittingStrategy(BaseModel):
    """Strategy for document splitting"""

    allow_uncategorized: bool = False


class SplitSettings(BaseModel):
    """Settings for document splitting."""

    splitting_strategy: SplittingStrategy = SplittingStrategy()


class SplitConfig(BaseModel):
    """Split configuration with categories and settings."""

    categories: list[SplitCategory] = []
    settings: SplitSettings = SplitSettings()


class ClassifyRule(BaseModel):
    """Classify rule, with type (rule target) and description (rule description)"""

    type: str
    description: str


class ClassifyParsingConfig(BaseModel):
    """Parsing config for Classify"""

    lang: str = Field(description="two-letter ISO 639 language code", default="en")
    max_pages: int | None = None
    target_pages: list[int] | None = None


class ClassifySettings(BaseModel):
    """Extra settings for Classify"""

    mode: Literal["FAST", "MULTIMODAL"] = "FAST"
    parsing_config: ClassifyParsingConfig = ClassifyParsingConfig()


class ClassifyConfig(BaseModel):
    """Classify configuration, with rules and settings"""

    rules: list[ClassifyRule] = []
    settings: ClassifySettings = ClassifySettings()


class ParseSettings(BaseModel):
    """Parsing settings for LlamaParse.

    See LlamaParse documentation for full options:
    /python/cloud/llamaparse/api-v2-guide/
    """

    tier: Literal["fast", "agentic"] = "agentic"
    version: str = "latest"
    lang: str | None = Field(
        default=None, description="Two-letter ISO 639 language code"
    )
    max_pages: int | None = None


class ParseConfig(BaseModel):
    """Parse configuration for LlamaParse."""

    settings: ParseSettings = ParseSettings()


class JsonSchema(BaseModel):
    """Pydantic wrapper for a JSON schema loaded via ResourceConfig."""

    type: str = "object"
    properties: dict[str, Any] = {}
    required: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for APIs that expect JSON schema."""
        return self.model_dump(exclude_none=True)


class Config(BaseModel):
    """Root configuration model for configs/config.json."""

    extract: ExtractConfig
    split: SplitConfig = SplitConfig()
    classify: ClassifyConfig = ClassifyConfig()
    parse: ParseConfig = ParseConfig()
