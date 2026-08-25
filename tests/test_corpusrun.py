"""The shared measurement loop, which is the comparison table's contract."""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

from corpusrun import run_corpus, validate_case_id_parity


def test_the_loop_warms_first_measures_each_case_and_writes_the_document(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cases = [
        {
            "id": "case00-clean",
            "image": "a.png",
            "lines": ["alpha beta", "line two"],
            "variant": "clean",
            "selection": {"model": "A"},
        },
        {
            "id": "case01-blur",
            "image": "b.png",
            "lines": ["gamma"],
            "variant": "blur",
            "selection": {"model": "B"},
        },
    ]
    seen = []

    def read(path, _case) -> list[str]:
        seen.append(path.name)
        return {"a.png": ["alpha beta", "line two"], "b.png": ["wrong"]}[path.name]

    out = tmp_path / "results-fake.json"
    rows = run_corpus(
        corpus,
        cases,
        read,
        tag="fake/engine",
        device="Test GPU",
        out=out,
        selector=lambda case: case["selection"],
    )

    # The first image for each declared selection was read once extra.
    assert seen == ["a.png", "b.png", "a.png", "b.png"]
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["engine"] == "fake/engine"
    assert document["device"] == "Test GPU"
    assert document["rows"] == rows
    # The row schema compare_engines.py consumes, in full.
    assert set(rows[0]) == {
        "id",
        "variant",
        "ms",
        "truth_lines",
        "observed_lines",
        "selection",
        "char_distance",
        "char_length",
        "word_distance",
        "word_length",
        "cer",
        "wer",
        "exact",
    }
    assert rows[0]["truth_lines"] == ["alpha beta", "line two"]
    assert rows[0]["observed_lines"] == ["alpha beta", "line two"]
    assert rows[0]["selection"] == {"model": "A"}
    assert rows[0]["exact"] is True and rows[1]["exact"] is False
    printed = capsys.readouterr().out
    assert "case00-clean" in printed and "cer=" in printed


def test_an_empty_results_document_is_refused_by_engine_name(tmp_path):

    import subprocess

    (tmp_path / "results-empty.json").write_text(
        json.dumps({"engine": "went/nowhere", "device": "CPU", "rows": []}), encoding="utf-8"
    )
    manifest = tmp_path / "ground-truth.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cases": [{"id": "expected", "variant": "clean"}],
            }
        ),
        encoding="utf-8",
    )
    script = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "compare_engines.py"
    done = subprocess.run(
        [sys.executable, str(script), str(tmp_path), str(manifest)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 2
    assert "went/nowhere" in done.stderr
    assert "missing ids ['expected']" in done.stderr


def test_result_case_id_parity_refuses_missing_and_extra_rows():
    cases = [{"id": "one"}, {"id": "two"}]
    rows = [{"id": "one"}, {"id": "ghost"}]

    with pytest.raises(ValueError, match="missing \\['two'\\].*extra \\['ghost'\\]"):
        validate_case_id_parity(cases, rows)
