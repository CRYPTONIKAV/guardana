"""The hooks demo rule proves its contract the way a run and a plan read it.

`not_applicable_to` must skip the rule as `not_applicable` (never as a check
that ran), and `estimated_requests_for` must price the rule per target.
"""

from guardana.core.plan import build_plan
from guardana.core.profile import default_profile
from guardana.core.registry import Registry
from guardana.core.target import ArtifactTarget
from guardana.core.testing import files_target
from guardana_reference_pack.hooks_demo import MarkerHookRule


def _registry() -> Registry:
    registry = Registry()
    registry.register_rule(MarkerHookRule())
    return registry


def test_a_target_with_no_text_files_skips_the_rule_as_not_applicable() -> None:
    target = files_target({"app.py": "print('hello')\n"})
    assert (
        MarkerHookRule().not_applicable_to(target) == "target lists no .txt files"
    )

    plan = build_plan(_registry(), default_profile(), target)

    assert plan.skipped_rule_ids == ("reference.demo.marker_hook",)


def test_a_target_with_text_files_applies_the_rule_and_costs_nothing() -> None:
    target = files_target({"notes.txt": "hello\n"})
    assert MarkerHookRule().not_applicable_to(target) is None

    plan = build_plan(_registry(), default_profile(), target)

    assert plan.rules == ("reference.demo.marker_hook",)
    assert plan.unknown_cost == ()
    assert plan.is_complete
    assert (plan.min_requests, plan.max_requests) == (0, 0)


def test_the_hook_prices_this_target_not_the_declaration() -> None:
    target = files_target({"notes.txt": "hello\n"})
    rule = MarkerHookRule()

    assert rule.estimated_requests == 0
    assert rule.estimated_requests_for(target) == 0


def test_an_empty_directory_has_nothing_to_check(tmp_path_factory) -> None:
    target = ArtifactTarget(tmp_path_factory.mktemp("empty"))
    assert (
        MarkerHookRule().not_applicable_to(target) == "target lists no .txt files"
    )
