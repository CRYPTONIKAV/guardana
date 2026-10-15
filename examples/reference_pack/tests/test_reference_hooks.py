"""The hooks demo rule honors the contract a run and a plan rely on.

`not_applicable_to` returns why there is nothing to check (never a silent
pass), and `estimated_requests_for` prices this target (an honest zero: the
rule only reads local files). The engine side — recording the skip as
`not_applicable` in the run and the plan, pricing with the hook — is pinned
upstream (`test_seeded_target.py`, `test_plan_static_cost.py`); what this
module pins is that this rule speaks that contract. All imports stay on the
pack's supported surface (see `test_reference_surface.py`).
"""

from guardana.core.testing import files_target
from guardana_reference_pack.hooks_demo import MarkerHookRule


def test_a_target_with_no_text_files_has_nothing_to_check() -> None:
    target = files_target({"app.py": "print('hello')\n"})

    assert MarkerHookRule().not_applicable_to(target) == "target lists no .txt files"


def test_a_target_with_text_files_applies_the_rule() -> None:
    target = files_target({"notes.txt": "hello\n"})

    assert MarkerHookRule().not_applicable_to(target) is None


def test_the_rule_costs_nothing_on_any_target_because_it_only_reads_files() -> None:
    target = files_target({"notes.txt": "hello\n"})
    rule = MarkerHookRule()

    assert rule.estimated_requests == 0
    assert rule.estimated_requests_for(target) == 0
