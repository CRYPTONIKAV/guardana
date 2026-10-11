"""security-insights.yml stays valid against the OpenSSF spec v2.2.0 essentials.

The spec's machine-readable schema is CUE (spec/schema.cue), which this
repository does not run; this test mirrors v2.2.0 for the sections the file
uses (required keys and allowed keys per section, plus the format rules the
CUE schema enforces: semver, YYYY-MM-DD dates, URL and email shapes), so a
missing or unknown field fails here instead of in a scanner.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import yaml

_ROOT = Path(__file__).resolve().parents[2]
DOC = yaml.safe_load((_ROOT / "security-insights.yml").read_text(encoding="utf-8"))

_URL = re.compile(r"^https?://[^\s]+$")
_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SEMVER = re.compile(r"^[1-9]+\.[0-9]+\.[0-9]+$")

# Section shapes mirrored from spec v2.2.0 for what this file uses:
# every required key must be present, and no key outside the allowed set
# may appear (the CUE definitions are closed).
_REQUIRED_TOP = {"header", "project", "repository"}
_ALLOWED_TOP = {"header", "project", "repository"}

_REQUIRED_HEADER = {"schema-version", "last-updated", "last-reviewed", "url"}
_ALLOWED_HEADER = _REQUIRED_HEADER | {"comment"}

_REQUIRED_PROJECT = {
    "name",
    "homepage",
    "administrators",
    "documentation",
    "repositories",
    "vulnerability-reporting",
}
_ALLOWED_PROJECT = _REQUIRED_PROJECT
_ALLOWED_PROJECT_DOCUMENTATION = {"code-of-conduct", "release-process"}
_REQUIRED_REPORTING = {"reports-accepted", "contact", "policy"}
_ALLOWED_REPORTING = _REQUIRED_REPORTING | {"bug-bounty-available", "comment"}

_REQUIRED_REPOSITORY = {"url", "status", "core-team", "documentation", "license", "release"}
_ALLOWED_REPOSITORY = _REQUIRED_REPOSITORY | {
    "accepts-change-request",
    "accepts-automated-change-request",
    "security",
}
_ALLOWED_REPOSITORY_DOCUMENTATION = {
    "contributing-guide",
    "review-policy",
    "governance",
    "security-policy",
}
_REQUIRED_LICENSE = {"url", "expression"}
_ALLOWED_LICENSE = _REQUIRED_LICENSE
_REQUIRED_RELEASE = {"changelog", "automated-pipeline", "attestations", "distribution-points"}
_ALLOWED_RELEASE = _REQUIRED_RELEASE

_PYPI_DISTRIBUTIONS = {
    "pkg:pypi/guardana-core",
    "pkg:pypi/guardana-rules",
    "pkg:pypi/guardana-report",
    "pkg:pypi/guardana-cli",
    "pkg:pypi/guardana-server",
}
_CONTAINER_IMAGES = {
    "ghcr.io/guardana/guardana",
    "ghcr.io/guardana/guardana-collector",
}


def _github_url(url: str) -> bool:
    return urlparse(url).hostname == "github.com"


def test_top_level_sections_match_v2_2_0() -> None:
    assert set(DOC) >= _REQUIRED_TOP
    assert set(DOC) <= _ALLOWED_TOP
    header = DOC["header"]
    assert set(header) >= _REQUIRED_HEADER
    assert set(header) <= _ALLOWED_HEADER
    assert _SEMVER.match(str(header["schema-version"]))
    assert _DATE.match(header["last-updated"])
    assert _DATE.match(header["last-reviewed"])
    assert header["url"].endswith("/security-insights.yml")


def test_project_section_match_v2_2_0() -> None:
    project = DOC["project"]
    assert set(project) >= _REQUIRED_PROJECT
    assert set(project) <= _ALLOWED_PROJECT
    assert _github_url(project["homepage"])
    admins = project["administrators"]
    assert len(admins) >= 1
    for admin in admins:
        assert admin["name"]
        assert _EMAIL.match(admin["email"])
    documentation = project["documentation"]
    assert set(documentation) <= _ALLOWED_PROJECT_DOCUMENTATION
    for url in documentation.values():
        assert _URL.match(url)
    for repo in project["repositories"]:
        assert _github_url(repo["url"])


def test_vulnerability_reporting_matches_security_md() -> None:
    reporting = DOC["project"]["vulnerability-reporting"]
    assert set(reporting) >= _REQUIRED_REPORTING
    assert set(reporting) <= _ALLOWED_REPORTING
    assert reporting["reports-accepted"] is True
    contact = reporting["contact"]
    assert contact["primary"] is True
    assert _EMAIL.match(contact["email"])
    policy = reporting["policy"]
    assert policy.endswith("/SECURITY.md")
    security_md = (_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert contact["email"] in security_md


def test_repository_section_matches_v2_2_0() -> None:
    repository = DOC["repository"]
    assert set(repository) >= _REQUIRED_REPOSITORY
    assert set(repository) <= _ALLOWED_REPOSITORY
    assert _github_url(repository["url"])
    documentation = repository["documentation"]
    assert set(documentation) <= _ALLOWED_REPOSITORY_DOCUMENTATION
    assert documentation["review-policy"].endswith("CONTRIBUTING.md#commits-and-pull-requests")
    lic = repository["license"]
    assert set(lic) >= _REQUIRED_LICENSE
    assert set(lic) <= _ALLOWED_LICENSE
    assert lic["expression"] == "Apache-2.0"
    assert _URL.match(lic["url"])
    core = repository["core-team"]
    assert any(c.get("primary") and _EMAIL.match(c.get("email", "")) for c in core)


def test_release_artifacts_described() -> None:
    release = DOC["repository"]["release"]
    assert set(release) >= _REQUIRED_RELEASE
    assert set(release) <= _ALLOWED_RELEASE
    assert release["automated-pipeline"] is True
    predicates = {a["predicate-uri"] for a in release["attestations"]}
    assert "https://cyclonedx.org/bom" in predicates
    assert any("slsa.dev" in p for p in predicates)
    for attestation in release["attestations"]:
        assert _URL.match(attestation["location"])
    uris = [d["uri"] for d in release["distribution-points"]]
    assert set(uris) >= _PYPI_DISTRIBUTIONS
    assert set(uris) >= _CONTAINER_IMAGES
    assert any(urlparse(u).hostname == "github.com" and "releases" in u for u in uris)
