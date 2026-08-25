"""The shared measurement loop, which is the comparison table's contract."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

from corpusrun import run_corpus


def test_the_loop_warms_first_measures_each_case_and_writes_the_document(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cases = [
        {
            "id": "case00-clean",
            "image": "a.png",
            "lines": ["alpha beta", "line two"],
            "variant": "clean",
        },
        {"id": "case01-blur", "image": "b.png", "lines": ["gamma"], "variant": "blur"},
    ]
    seen = []

    def read(path) -> list[str]:
        seen.append(path.name)
        return {"a.png": ["alpha beta", "line two"], "b.png": ["wrong"]}[path.name]

    out = tmp_path / "results-fake.json"
    rows = run_corpus(corpus, cases, read, tag="fake/engine", device="Test GPU", out=out)

    # The first image was read once extra, before any timing.
    assert seen == ["a.png", "a.png", "b.png"]
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
    assert rows[0]["exact"] is True and rows[1]["exact"] is False
    printed = capsys.readouterr().out
    assert "case00-clean" in printed and "cer=" in printed


def test_an_empty_results_document_is_named_not_a_crash(tmp_path):
    """The guard runs before the arithmetic it guards (VOCR-0054)."""

    import subprocess

    (tmp_path / "results-empty.json").write_text(
        json.dumps({"engine": "went/nowhere", "device": "CPU", "rows": []}), encoding="utf-8"
    )
    script = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "compare_engines.py"
    done = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "went/nowhere" in done.stdout
    assert "(no rows)" in done.stdout
