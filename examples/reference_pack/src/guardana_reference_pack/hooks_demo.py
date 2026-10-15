"""`reference.demo.marker_hook`: what the additive rule hooks look like in one place.

A rule that reads every `.txt` file a target lists and reports lines starting
with `FLAG:`. It demonstrates the two hooks `docs/extending.md` documents for
rules, so a reader can see the contract without leaving the reference pack:

- `not_applicable_to` returns why there is nothing to check (a non-empty
  string), or `None` when the rule applies. The run and the plan record the
  rule as skipped `not_applicable`, never as a check that ran.
- `estimated_requests_for` prices the rule against one target. This rule only
  reads local files, so the honest price is zero — declared here, by the
  rule's author, never guessed by the engine (see
  `test_plan_static_cost.py` upstream for why the engine must not do that).
"""

from collections.abc import Iterable
from pathlib import Path

from guardana.core import (
    Capability,
    Evidence,
    Finding,
    Rule,
    RuleContext,
    RuleMeta,
    Severity,
    Target,
    TargetKind,
    Verdict,
)
from guardana.core.rule.fixture import FixtureOutcome, RuleFixture
from guardana.core.target.protocols import FileReader
from guardana.core.testing import files_target

SUFFIX = ".txt"


def _is_text_file(path: Path) -> bool:
    """Whether `path` is one this rule reads."""
    return path.suffix.lower() == SUFFIX


class MarkerHookRule(Rule):
    """Report `FLAG:` lines in `.txt` files; nothing else is a finding."""

    meta = RuleMeta(
        id="reference.demo.marker_hook",
        title="A text file carries a flagged line",
        severity=Severity.LOW,
        target_kind=TargetKind.ARTIFACT,
        required_capabilities=frozenset({Capability.READ_FILES}),
    )

    @property
    def estimated_requests(self) -> int | None:
        """Zero: this rule only reads local files, so it sends nothing anywhere."""
        return 0

    def estimated_requests_for(self, target: Target) -> int | None:
        """Zero for every target, priced per target: local reads cost no requests."""
        return 0

    def not_applicable_to(self, target: Target) -> str | None:
        """Say why there is nothing to check, or `None` when the rule applies."""
        if not isinstance(target, FileReader):
            return f"needs a file target, got {type(target).__name__}"
        for path in target.iter_files((SUFFIX,)):
            if _is_text_file(path):
                return None
        return "target lists no .txt files"

    def run(self, target: Target, ctx: RuleContext) -> Iterable[Finding]:
        """Grade each `.txt` file line by line; unreadable bytes are declined."""
        if not isinstance(target, FileReader):
            return
        for path in target.iter_files((SUFFIX,)):
            if not _is_text_file(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                yield self._declined(path, f"could not be read: {exc}")
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if line.startswith("FLAG:"):
                    yield self._flagged(path, number, line)

    def _flagged(self, path: Path, number: int, line: str) -> Finding:
        return Finding(
            rule_id=self.meta.id,
            severity=self.meta.severity,
            title=self.meta.title,
            taxonomy=self.meta.taxonomy,
            target_ref=str(path),
            evidence=Evidence(
                summary=f"flagged line: {line!r}",
                detail=f"file={path.name}; line={number}",
            ),
        )

    def _declined(self, path: Path, reason: str) -> Finding:
        return Finding(
            rule_id=self.meta.id,
            severity=self.meta.severity,
            title=self.meta.title,
            taxonomy=self.meta.taxonomy,
            target_ref=str(path),
            evidence=Evidence(summary=reason, detail=f"file={path.name}"),
            verdict=Verdict("inconclusive", 0.0, reason, self.meta.id),
        )

    def fixtures(self) -> Iterable[RuleFixture]:
        """Three samples: a flag, a clean file, and bytes that do not decode."""
        return (
            RuleFixture(
                "a text file with a flagged line",
                files_target({"notes.txt": "hello\nFLAG: review me\n"}),
                FixtureOutcome.FINDING,
            ),
            RuleFixture(
                "a text file with nothing flagged",
                files_target({"notes.txt": "hello\n"}),
                FixtureOutcome.CLEAN,
            ),
            RuleFixture(
                "a text file that is not UTF-8",
                files_target({"notes.txt": b"FLAG: \xff\xfe\n"}),
                FixtureOutcome.INCONCLUSIVE,
                note="the bytes do not decode, so no line is graded",
            ),
        )
