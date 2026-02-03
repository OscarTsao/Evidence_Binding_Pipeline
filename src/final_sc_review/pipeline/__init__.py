"""Pipeline package for Evidence Binding Pipeline.

Provides:
- ZooPipeline: Main retrieval pipeline using NV-Embed-v2 + Jina-Reranker-v3
- Config loading with validation
"""

from final_sc_review.pipeline.zoo_pipeline import (
    ZooPipeline,
    ZooPipelineConfig,
    load_zoo_pipeline_from_config,
    ConfigValidationError,
    VALID_RETRIEVERS,
    VALID_RERANKERS,
)

__all__ = [
    "ZooPipeline",
    "ZooPipelineConfig",
    "load_zoo_pipeline_from_config",
    "ConfigValidationError",
    "VALID_RETRIEVERS",
    "VALID_RERANKERS",
]
