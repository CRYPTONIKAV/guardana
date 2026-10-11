import time
from pathlib import Path

import pytest
from guardana.core.plugins import PluginMode, PluginTrust
from guardana.core.profile.model import Policy, Profile
from guardana.core.registry import Registry
from guardana.core.report import ShortfallKind
from guardana.core.rule import RuleContext
from guardana.core.runner import Runner
from guardana.core.target import ArtifactTarget
from guardana.rules.supply_chain import saved_model_ops
from guardana.rules.supply_chain.saved_model_ops import SavedModelOpsRule


def _findings(tmp_path: Path) -> list[tuple[str, str]]:
    rule = SavedModelOpsRule()
    return [
        (f.severity.name, f.evidence.summary)
        for f in rule.run(ArtifactTarget(tmp_path), RuleContext())
    ]


def _unread(root: Path) -> list[str]:
    """The files the rule names as unread components when run over `root`."""
    ctx = RuleContext()
    list(SavedModelOpsRule().run(ArtifactTarget(root), ctx))
    return [
        gap.name
        for gap in ctx.shortfalls()
        if gap.kind is ShortfallKind.UNEXAMINED_COMPONENT
        and gap.detail.startswith("guardana.supply_chain.saved_model_ops could not read it: ")
    ]


def test_flags_readfile_op(tmp_path: Path) -> None:
    (tmp_path / "saved_model.pb").write_bytes(b"\x08\x01tensorflow...ReadFile...\x00")
    findings = _findings(tmp_path)
    assert len(findings) == 1
    assert findings[0][0] == "MEDIUM"
    assert "ReadFile" in findings[0][1]


def test_flags_writefile_op(tmp_path: Path) -> None:
    (tmp_path / "saved_model.pb").write_bytes(b"tensorflow graph WriteFile op here")
    assert any("WriteFile" in s for _, s in _findings(tmp_path))


def test_reports_each_dangerous_op_once(tmp_path: Path) -> None:
    (tmp_path / "saved_model.pb").write_bytes(b"tensorflow ReadFile ... WriteFile ... ReadFile")
    ops = sorted(s.split()[0] for _, s in _findings(tmp_path))
    assert ops == ["ReadFile", "WriteFile"]


def test_clean_graph_not_flagged(tmp_path: Path) -> None:
    (tmp_path / "saved_model.pb").write_bytes(b"tensorflow MatMul BiasAdd Relu Softmax")
    assert _findings(tmp_path) == []


def test_non_tf_pb_without_ops_is_clean(tmp_path: Path) -> None:
    (tmp_path / "data.pb").write_bytes(b"some other protobuf payload")
    assert _findings(tmp_path) == []


def test_a_graph_past_the_read_bound_is_unverified_and_a_named_shortfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(saved_model_ops, "_MAX_SCAN_BYTES", 64)
    (tmp_path / "saved_model.pb").write_bytes(b"\x08\x01" + b"\x00" * 128 + b"WriteFile")

    findings = list(SavedModelOpsRule().run(ArtifactTarget(tmp_path), RuleContext()))

    assert [f.title for f in findings] == ["SavedModel not scanned"]
    assert findings[0].verdict is not None
    assert findings[0].verdict.outcome == "inconclusive"
    assert _unread(tmp_path) == [str(tmp_path / "saved_model.pb")]


def test_a_graph_read_whole_is_no_shortfall(tmp_path: Path) -> None:
    (tmp_path / "saved_model.pb").write_bytes(b"\x08\x01 WriteFile")

    assert _unread(tmp_path) == []


def test_an_unreadable_graph_is_unverified_and_a_named_shortfall(tmp_path: Path) -> None:
    path = tmp_path / "saved_model.pb"
    path.write_bytes(b"\x08\x01 WriteFile")
    path.chmod(0)
    ctx = RuleContext()
    try:
        findings = list(SavedModelOpsRule().run(ArtifactTarget(tmp_path), ctx))
    finally:
        path.chmod(0o600)

    assert [(f.title, f.evidence.summary) for f in findings] == [
        ("SavedModel not scanned", "the file could not be read")
    ]
    assert findings[0].verdict is not None
    assert findings[0].verdict.outcome == "inconclusive"
    assert [gap.name for gap in ctx.shortfalls()] == [str(path)]


def _graph_text(op_field: str) -> str:
    return (
        'meta_graphs {\n  graph_def {\n    node {\n      name: "save/Write"\n'
        f"      {op_field}\n    }}\n  }}\n}}\n"
    )


@pytest.mark.parametrize(
    "op_field",
    [
        'op: "WriteFile"',
        "op: 'ReadFile'",
        'op: "Write" "File"',
        'op: "Writ\\145File"',
        'op: "\\x57riteFile"',
        'op : # the operator\n  "WriteFile"',
    ],
    ids=["double-quoted", "single-quoted", "concatenated", "octal-escape", "hex-escape", "comment"],
)
def test_a_filesystem_op_in_a_text_savedmodel_is_flagged(tmp_path: Path, op_field: str) -> None:
    (tmp_path / "saved_model.pbtxt").write_text(_graph_text(op_field))
    findings = _findings(tmp_path)
    assert [sev for sev, _ in findings] == ["MEDIUM"]
    assert findings[0][1].split()[0] in {"ReadFile", "WriteFile"}


def test_a_text_savedmodel_without_file_ops_is_clean_and_examined(tmp_path: Path) -> None:
    (tmp_path / "saved_model.pbtxt").write_text(
        _graph_text('op: "MatMul"') + '# a WriteFile op would be flagged\nname: "ReadFile"\n'
    )
    ctx = RuleContext()
    assert list(SavedModelOpsRule().run(ArtifactTarget(tmp_path), ctx)) == []
    assert ctx.examined_paths() == {str(tmp_path / "saved_model.pbtxt")}


def test_another_text_protobuf_is_not_read_as_a_savedmodel(tmp_path: Path) -> None:
    (tmp_path / "label_map.pbtxt").write_text('item { op: "WriteFile" }\n')
    assert _findings(tmp_path) == []


def test_an_unreadable_text_savedmodel_is_unverified_and_a_named_shortfall(
    tmp_path: Path,
) -> None:
    path = tmp_path / "saved_model.pbtxt"
    path.write_text(_graph_text('op: "WriteFile"'))
    path.chmod(0)
    ctx = RuleContext()
    try:
        findings = list(SavedModelOpsRule().run(ArtifactTarget(tmp_path), ctx))
    finally:
        path.chmod(0o600)

    assert [f.title for f in findings] == ["SavedModel not scanned"]
    assert [gap.name for gap in ctx.shortfalls()] == [str(path)]


def test_a_text_savedmodel_past_the_read_bound_is_a_named_shortfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(saved_model_ops, "_MAX_SCAN_BYTES", 64)
    (tmp_path / "saved_model.pbtxt").write_text("#" * 128 + "\n" + _graph_text('op: "WriteFile"'))

    findings = list(SavedModelOpsRule().run(ArtifactTarget(tmp_path), RuleContext()))

    assert [f.title for f in findings] == ["SavedModel not scanned"]
    assert _unread(tmp_path) == [str(tmp_path / "saved_model.pbtxt")]


def test_a_graph_read_whole_is_examined(tmp_path: Path) -> None:
    (tmp_path / "saved_model.pb").write_bytes(b"\n\x05const\x12\x05Const")
    ctx = RuleContext()
    assert list(SavedModelOpsRule().run(ArtifactTarget(tmp_path), ctx)) == []
    assert ctx.examined_paths() == {str(tmp_path / "saved_model.pb")}


@pytest.mark.parametrize("name", ["saved_model.pb", "saved_model.pbtxt"])
def test_a_savedmodel_no_rule_reads_is_a_coverage_shortfall(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"\x08\x01")
    profile = Profile(name="t", policy=Policy(exclude=("guardana.supply_chain.saved_model_ops",)))
    result = Runner(
        registry=Registry.discover(PluginTrust(mode=PluginMode.BUILTINS)), profile=profile
    ).run(ArtifactTarget(tmp_path))

    assert [gap.name for gap in result.coverage_shortfall] == ["tensorflow-savedmodel"]


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("saved_model.pb", b"\n\x05const\x12\x05Const"),
        ("saved_model.pbtxt", b'node { op: "Const" }\n'),
    ],
)
def test_a_savedmodel_without_file_ops_is_read_and_leaves_no_shortfall(
    tmp_path: Path, name: str, content: bytes
) -> None:
    (tmp_path / name).write_bytes(content)
    result = Runner(
        registry=Registry.discover(PluginTrust(mode=PluginMode.BUILTINS)),
        profile=Profile(name="t", policy=Policy()),
    ).run(ArtifactTarget(tmp_path))

    assert result.coverage_shortfall == ()
    assert result.findings == ()


@pytest.mark.parametrize(
    "tail",
    [b"#" * 4096, b"# x " * 4096, b'"a" ' * 4096 + b"#" * 4096],
    ids=["comment-run", "comment-words", "strings-then-comment"],
)
def test_a_crafted_text_graph_is_read_in_bounded_time(tmp_path: Path, tail: bytes) -> None:
    (tmp_path / "saved_model.pbtxt").write_bytes(b"node { op: " + tail)

    started = time.monotonic()
    assert _findings(tmp_path) == []
    assert time.monotonic() - started < 5.0


@pytest.mark.parametrize(
    "op_field",
    [
        'op:\x1c"WriteFile"',
        'op\xa0: "WriteFile"',
        'op:　"WriteFile"',
        'op : "WriteFile"',
        'op: "\\N{LATIN CAPITAL LETTER W}riteFile"',
        'op: "Write\\u0046ile"',
    ],
    ids=["info-separator", "no-break-space", "ideographic-space", "space", "named", "unicode"],
)
def test_a_spelling_protobuf_parses_as_a_filesystem_op_is_flagged(
    tmp_path: Path, op_field: str
) -> None:
    (tmp_path / "saved_model.pbtxt").write_text(_graph_text(op_field), encoding="utf-8")
    assert [sev for sev, _ in _findings(tmp_path)] == ["MEDIUM"]


@pytest.mark.parametrize(
    "op_field",
    ['op: "\\N{NO SUCH CHARACTER NAME}"', 'op: "Write\\u12File"'],
    ids=["unknown-name", "short-unicode"],
)
def test_an_op_value_that_cannot_be_decoded_is_unscanned(tmp_path: Path, op_field: str) -> None:
    (tmp_path / "saved_model.pbtxt").write_text(_graph_text(op_field), encoding="utf-8")
    ctx = RuleContext()
    findings = list(SavedModelOpsRule().run(ArtifactTarget(tmp_path), ctx))
    assert [f.title for f in findings] == ["SavedModel not scanned"]
    assert [gap.name for gap in ctx.shortfalls()] == [str(tmp_path / "saved_model.pbtxt")]


def test_an_ordinary_op_with_unicode_spacing_is_clean(tmp_path: Path) -> None:
    (tmp_path / "saved_model.pbtxt").write_text(_graph_text('op:\xa0"MatMul"'), encoding="utf-8")
    assert _findings(tmp_path) == []
