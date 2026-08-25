"""Versioned corpus-manifest I/O and JSON Schema validation."""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 4
SCHEMA_PATH = Path(__file__).with_name("corpus-case.schema.json")


def load_manifest(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: expected corpus schema version {SCHEMA_VERSION}")
    if not isinstance(document.get("cases"), list):
        raise ValueError(f"{path}: cases must be an array")
    return document


def load_cases(path: Path) -> list[dict]:
    return load_manifest(path)["cases"]


def require_engine_selections(cases: list[dict], engine: str) -> list[dict]:
    missing = [
        case["id"]
        for case in cases
        if not isinstance(case.get("recognition"), dict) or case["recognition"].get(engine) is None
    ]
    if missing:
        raise ValueError(f"no {engine} model is declared for cases {missing}")
    return [case["recognition"][engine] for case in cases]


def validate_manifest(document: dict) -> None:
    try:
        from jsonschema import Draft202012Validator  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - dependency installation failure
        raise RuntimeError("manifest validation requires the corpus or dev extra") from error
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document), key=lambda item: list(item.path)
    )
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "manifest"
        raise ValueError(f"invalid corpus manifest at {location}: {error.message}")
    ids = [case["id"] for case in document["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("invalid corpus manifest: case ids must be unique")


def write_manifest(path: Path, document: dict) -> None:
    validate_manifest(document)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "load_cases",
    "load_manifest",
    "require_engine_selections",
    "validate_manifest",
    "write_manifest",
]
