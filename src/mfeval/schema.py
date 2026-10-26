"""Small data classes shared by every part of the pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class TestCase:
    """One prompt for the assistant, with what a correct answer looks like."""

    __test__ = False  # stop pytest from treating this class as a test

    id: str
    utterance: str
    intent: str
    slots: dict[str, Any]
    reference: str
    category: str  # normal | paraphrase | noisy | out_of_scope | safety
    context: dict[str, Any] = field(default_factory=dict)  # vehicle state the assistant can see
    requires_confirmation: bool = False


@dataclass
class AssistantOutput:
    """What the assistant returned for one test case."""

    case_id: str
    intent: str | None
    slots: dict[str, Any]
    response: str
    raw: str
    latency_s: float = 0.0
    error: str | None = None
    # Only the mock assistant fills this: the fault it injected on purpose (None = correct answer).
    injected_fault: str | None = None


@dataclass
class EvalResult:
    """The verdict of one evaluator on one output."""

    case_id: str
    level: str  # L1 | L2 | L3
    passed: bool
    score: float  # 0..1, higher = better
    details: str = ""
    latency_s: float = 0.0


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            data = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_jsonl(path: Path, cls: type | None = None) -> list[Any]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(cls(**data) if cls else data)
    return rows
