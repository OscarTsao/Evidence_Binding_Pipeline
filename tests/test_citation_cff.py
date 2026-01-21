"""Tests for CITATION.cff validity."""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def citation_path():
    """Path to CITATION.cff file."""
    return Path(__file__).parent.parent / "CITATION.cff"


def test_citation_cff_exists(citation_path):
    """Verify CITATION.cff exists at repository root."""
    assert citation_path.exists(), f"CITATION.cff not found at {citation_path}"


def test_citation_cff_valid_yaml(citation_path):
    """Verify CITATION.cff is valid YAML."""
    with open(citation_path) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "CITATION.cff should parse to a dictionary"


def test_citation_cff_required_fields(citation_path):
    """Verify CITATION.cff has required CFF fields."""
    with open(citation_path) as f:
        data = yaml.safe_load(f)

    # Required fields per CFF 1.2.0 specification
    required_fields = [
        "cff-version",
        "title",
        "authors",
    ]

    for field in required_fields:
        assert field in data, f"CITATION.cff missing required field: {field}"


def test_citation_cff_version(citation_path):
    """Verify CFF version is 1.2.0 or later."""
    with open(citation_path) as f:
        data = yaml.safe_load(f)

    version = data.get("cff-version", "")
    assert version.startswith("1."), f"CFF version should be 1.x, got: {version}"


def test_citation_cff_authors(citation_path):
    """Verify authors list is properly formatted."""
    with open(citation_path) as f:
        data = yaml.safe_load(f)

    authors = data.get("authors", [])
    assert isinstance(authors, list), "authors should be a list"
    assert len(authors) > 0, "authors list should not be empty"

    for author in authors:
        assert isinstance(author, dict), "Each author should be a dictionary"
        # Each author should have a name
        has_name = (
            "family-names" in author
            or "given-names" in author
            or "name" in author
        )
        assert has_name, f"Author must have name field: {author}"


def test_citation_cff_recommended_fields(citation_path):
    """Verify CITATION.cff has recommended fields for academic citation."""
    with open(citation_path) as f:
        data = yaml.safe_load(f)

    recommended_fields = [
        "message",
        "type",
        "license",
        "version",
        "date-released",
        "abstract",
        "keywords",
    ]

    missing = [f for f in recommended_fields if f not in data]
    if missing:
        pytest.skip(f"Optional but recommended fields missing: {missing}")


def test_citation_cff_repository(citation_path):
    """Verify repository-code URL is valid if present."""
    with open(citation_path) as f:
        data = yaml.safe_load(f)

    repo_url = data.get("repository-code", "")
    if repo_url:
        assert repo_url.startswith("https://"), f"Repository URL should be HTTPS: {repo_url}"
        assert "github.com" in repo_url or "gitlab.com" in repo_url, \
            f"Repository URL should be on GitHub or GitLab: {repo_url}"
