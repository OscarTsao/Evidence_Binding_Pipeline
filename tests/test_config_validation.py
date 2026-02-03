"""Tests for configuration validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from final_sc_review.pipeline.zoo_pipeline import (
    ConfigValidationError,
    VALID_RETRIEVERS,
    VALID_RERANKERS,
)


class TestConfigValidation:
    """Tests for config validation in zoo pipeline."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create temporary config directory."""
        return tmp_path

    def test_valid_retrievers_list(self):
        """Test VALID_RETRIEVERS contains expected model."""
        assert "nv-embed-v2" in VALID_RETRIEVERS

    def test_valid_rerankers_list(self):
        """Test VALID_RERANKERS contains expected model."""
        assert "jina-reranker-v3" in VALID_RERANKERS

    def test_config_file_not_found(self, temp_config_dir):
        """Test error when config file doesn't exist."""
        from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config

        non_existent = temp_config_dir / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError) as exc_info:
            load_zoo_pipeline_from_config(non_existent)
        assert "not found" in str(exc_info.value).lower()

    def test_empty_config(self, temp_config_dir):
        """Test error on empty config file."""
        from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config

        config_path = temp_config_dir / "empty.yaml"
        config_path.write_text("")

        with pytest.raises(ConfigValidationError) as exc_info:
            load_zoo_pipeline_from_config(config_path)
        assert "empty" in str(exc_info.value).lower()

    def test_missing_paths_section(self, temp_config_dir):
        """Test error when paths section is missing."""
        from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config

        config_path = temp_config_dir / "no_paths.yaml"
        config_path.write_text(yaml.dump({"models": {"retriever_name": "nv-embed-v2"}}))

        with pytest.raises(ConfigValidationError) as exc_info:
            load_zoo_pipeline_from_config(config_path)
        assert "paths" in str(exc_info.value).lower()

    def test_missing_required_path_keys(self, temp_config_dir):
        """Test error when required path keys are missing."""
        from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config

        config_path = temp_config_dir / "missing_corpus.yaml"
        config_path.write_text(yaml.dump({
            "paths": {
                "cache_dir": "/tmp/cache"
                # Missing sentence_corpus
            }
        }))

        with pytest.raises(ConfigValidationError) as exc_info:
            load_zoo_pipeline_from_config(config_path)
        assert "sentence_corpus" in str(exc_info.value)

    def test_invalid_retriever_name(self, temp_config_dir):
        """Test error on invalid retriever name."""
        from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config

        config_path = temp_config_dir / "bad_retriever.yaml"
        config_path.write_text(yaml.dump({
            "paths": {
                "sentence_corpus": "data/corpus.jsonl",
                "cache_dir": "/tmp/cache"
            },
            "models": {
                "retriever_name": "invalid-retriever"
            }
        }))

        with pytest.raises(ConfigValidationError) as exc_info:
            load_zoo_pipeline_from_config(config_path)
        assert "invalid-retriever" in str(exc_info.value)
        assert "Valid options" in str(exc_info.value)

    def test_invalid_reranker_name(self, temp_config_dir):
        """Test error on invalid reranker name."""
        from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config

        config_path = temp_config_dir / "bad_reranker.yaml"
        config_path.write_text(yaml.dump({
            "paths": {
                "sentence_corpus": "data/corpus.jsonl",
                "cache_dir": "/tmp/cache"
            },
            "models": {
                "retriever_name": "nv-embed-v2",
                "reranker_name": "invalid-reranker"
            }
        }))

        with pytest.raises(ConfigValidationError) as exc_info:
            load_zoo_pipeline_from_config(config_path)
        assert "invalid-reranker" in str(exc_info.value)

    def test_negative_top_k_retriever(self, temp_config_dir):
        """Test error on negative top_k_retriever."""
        from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config

        config_path = temp_config_dir / "negative_k.yaml"
        config_path.write_text(yaml.dump({
            "paths": {
                "sentence_corpus": "data/corpus.jsonl",
                "cache_dir": "/tmp/cache"
            },
            "retriever": {
                "top_k_retriever": -1
            }
        }))

        with pytest.raises(ConfigValidationError) as exc_info:
            load_zoo_pipeline_from_config(config_path)
        assert "top_k_retriever" in str(exc_info.value)
        assert "positive" in str(exc_info.value)

    def test_top_k_final_exceeds_retriever(self, temp_config_dir):
        """Test error when top_k_final > top_k_retriever."""
        from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config

        config_path = temp_config_dir / "k_mismatch.yaml"
        config_path.write_text(yaml.dump({
            "paths": {
                "sentence_corpus": "data/corpus.jsonl",
                "cache_dir": "/tmp/cache"
            },
            "retriever": {
                "top_k_retriever": 10,
                "top_k_final": 20
            }
        }))

        with pytest.raises(ConfigValidationError) as exc_info:
            load_zoo_pipeline_from_config(config_path)
        assert "top_k_final" in str(exc_info.value)
        assert "cannot exceed" in str(exc_info.value)

    def test_zero_rrf_k(self, temp_config_dir):
        """Test error on zero rrf_k."""
        from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config

        config_path = temp_config_dir / "zero_rrf.yaml"
        config_path.write_text(yaml.dump({
            "paths": {
                "sentence_corpus": "data/corpus.jsonl",
                "cache_dir": "/tmp/cache"
            },
            "retriever": {
                "rrf_k": 0
            }
        }))

        with pytest.raises(ConfigValidationError) as exc_info:
            load_zoo_pipeline_from_config(config_path)
        assert "rrf_k" in str(exc_info.value)
        assert "positive" in str(exc_info.value)
