"""CLI option wiring without loading a model or touching a GPU."""

from types import SimpleNamespace

from vulkanocr import InferenceOptions, cli


def test_precision_defaults_to_fp32():
    arguments = cli._arguments(["page.png"])
    assert arguments.precision == "fp32"


def test_precision_choice_reaches_the_single_engine(monkeypatch):
    received = []
    monkeypatch.setattr(cli, "models_for", lambda _name: object())
    monkeypatch.setattr(
        cli,
        "OcrEngine",
        lambda _models, *, options: received.append(options) or SimpleNamespace(),
    )

    cli._reader(cli._arguments(["page.png", "--precision", "int8"]))

    assert received == [InferenceOptions.int8()]
