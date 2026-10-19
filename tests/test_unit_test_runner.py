"""Contract tests for the clean-clone pytest/coverage wrapper itself."""

from __future__ import annotations

import argparse
from pathlib import Path

import run_unit_tests as runner


def test_pytest_arguments_select_coverage_html_floor_and_extras() -> None:
    plain = runner.pytest_arguments(coverage=False, html=False, extra=("-x",))
    assert plain[0] == "-q"
    assert "--cov=pkate" not in plain
    assert plain[-1] == "-x"

    covered = runner.pytest_arguments(coverage=True, html=False)
    assert "--cov=pkate" in covered
    assert "--cov-fail-under=80" in covered
    assert not any(value.startswith("--cov-report=html:") for value in covered)

    html = runner.pytest_arguments(coverage=True, html=True)
    assert "--cov-report=html:htmlcov" in html
    assert "tests/test_unit_test_runner.py" in html


def test_parse_args_retains_unknown_pytest_options(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_unit_tests.py", "--html", "-x", "--maxfail=2"])
    args, extra = runner.parse_args()
    assert args.html is True
    assert extra == ["-x", "--maxfail=2"]


def test_main_changes_to_project_and_dispatches_pytest(tmp_path: Path, monkeypatch) -> None:
    captured = {}
    args = argparse.Namespace(coverage=False, html=True)
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "parse_args", lambda: (args, ["-x"]))
    def fake_main(arguments):
        captured["arguments"] = arguments
        return 7
    monkeypatch.setattr(runner.pytest, "main", fake_main)
    assert runner.main() == 7
    assert Path.cwd() == tmp_path
    assert "--cov=pkate" in captured["arguments"]
    assert "--cov-report=html:htmlcov" in captured["arguments"]
    assert captured["arguments"][-1] == "-x"
