"""Dataset loading and deterministic baseline metrics for RAG answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REGRESSION_THRESHOLDS = {
    "pass_rate_drop": 0.05,
    "keyword_recall_drop": 0.05,
    "citation_validity_drop": 0.05,
    "latency_increase_ratio": 0.25,
}


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_answer: str = ""
    knowledge_base_ids: list[str] = field(default_factory=list)
    case_id: str = ""


def load_dataset(path: str | Path) -> list[EvaluationCase]:
    """Load a UTF-8 JSONL dataset and validate every case."""
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"评测数据集不存在: {dataset_path}")
    cases: list[EvaluationCase] = []
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"评测数据集第 {line_number} 行不是合法 JSON") from exc
            question = str(payload.get("question") or "").strip()
            if not question:
                raise ValueError(f"评测数据集第 {line_number} 行缺少 question")
            keywords = payload.get("expected_keywords") or []
            if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
                raise ValueError(f"评测数据集第 {line_number} 行 expected_keywords 必须是字符串数组")
            cases.append(EvaluationCase(
                case_id=str(payload.get("case_id") or line_number),
                question=question,
                expected_keywords=[item.strip() for item in keywords if item.strip()],
                expected_answer=str(payload.get("expected_answer") or "").strip(),
                knowledge_base_ids=[str(item) for item in payload.get("knowledge_base_ids") or []],
            ))
    if not cases:
        raise ValueError("评测数据集不能为空")
    return cases


def evaluate_answer(case: EvaluationCase, answer: str, sources: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Calculate stable metrics that do not require a second judge model."""
    normalized_answer = (answer or "").strip().lower()
    expected_keywords = [item.lower() for item in case.expected_keywords]
    matched_keywords = [item for item in expected_keywords if item in normalized_answer]
    keyword_recall = len(matched_keywords) / len(expected_keywords) if expected_keywords else 1.0
    source_list = list(sources)
    citation_numbers = {int(value) for value in re.findall(r"\[资料(\d+)\]", answer or "")}
    valid_citations = {number for number in citation_numbers if 1 <= number <= len(source_list)}
    citation_validity = len(valid_citations) / len(citation_numbers) if citation_numbers else 0.0
    has_answer = bool(normalized_answer)
    return {
        "case_id": case.case_id,
        "question": case.question,
        "has_answer": has_answer,
        "keyword_recall": round(keyword_recall, 4),
        "matched_keywords": matched_keywords,
        "expected_keywords": case.expected_keywords,
        "source_count": len(source_list),
        "citation_count": len(citation_numbers),
        "citation_validity": round(citation_validity, 4),
        "passed": has_answer and keyword_recall >= 0.8 and (not source_list or citation_validity >= 0.8),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"case_count": 0, "pass_rate": 0.0, "keyword_recall": 0.0, "citation_validity": 0.0}
    count = len(results)
    return {
        "case_count": count,
        "passed_count": sum(1 for item in results if item["passed"]),
        "pass_rate": round(sum(1 for item in results if item["passed"]) / count, 4),
        "keyword_recall": round(sum(item["keyword_recall"] for item in results) / count, 4),
        "citation_validity": round(sum(item["citation_validity"] for item in results) / count, 4),
        "average_latency_ms": round(sum(item.get("latency_ms", 0) for item in results) / count),
    }


def compare_evaluation_summaries(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare stable quality metrics and report threshold-based regressions."""
    limits = {**DEFAULT_REGRESSION_THRESHOLDS, **(thresholds or {})}
    quality_metrics = {
        "pass_rate": "pass_rate_drop",
        "keyword_recall": "keyword_recall_drop",
        "citation_validity": "citation_validity_drop",
    }
    deltas: dict[str, float] = {}
    regressions: list[dict[str, Any]] = []
    for metric, threshold_name in quality_metrics.items():
        candidate_value = float(candidate.get(metric) or 0)
        baseline_value = float(baseline.get(metric) or 0)
        delta = round(candidate_value - baseline_value, 4)
        deltas[metric] = delta
        allowed_drop = float(limits[threshold_name])
        if delta < -allowed_drop:
            regressions.append({
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
                "threshold": allowed_drop,
            })

    baseline_latency = float(baseline.get("average_latency_ms") or 0)
    candidate_latency = float(candidate.get("average_latency_ms") or 0)
    latency_ratio = round((candidate_latency - baseline_latency) / baseline_latency, 4) if baseline_latency > 0 else 0.0
    deltas["average_latency_ms"] = round(candidate_latency - baseline_latency, 2)
    deltas["latency_increase_ratio"] = latency_ratio
    latency_threshold = float(limits["latency_increase_ratio"])
    if baseline_latency > 0 and latency_ratio > latency_threshold:
        regressions.append({
            "metric": "average_latency_ms",
            "baseline": baseline_latency,
            "candidate": candidate_latency,
            "delta": deltas["average_latency_ms"],
            "increase_ratio": latency_ratio,
            "threshold": latency_threshold,
        })
    return {
        "passed": not regressions,
        "regressions": regressions,
        "deltas": deltas,
        "thresholds": limits,
    }
