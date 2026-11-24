from evaluation.rag_evaluator import (
    EvaluationCase,
    compare_evaluation_summaries,
    evaluate_answer,
    summarize_results,
)


def test_evaluate_answer_counts_keywords_and_valid_citations() -> None:
    case = EvaluationCase(case_id="1", question="配置？", expected_keywords=["内存", "硬盘"])
    result = evaluate_answer(case, "内存 16GB，硬盘 512GB。[资料1]", [{"content": "规格"}])
    assert result["keyword_recall"] == 1.0
    assert result["citation_validity"] == 1.0
    assert result["passed"] is True


def test_summarize_results_is_stable_for_empty_input() -> None:
    assert summarize_results([])["case_count"] == 0


def test_comparison_passes_changes_within_thresholds() -> None:
    baseline = {"pass_rate": 0.9, "keyword_recall": 0.9, "citation_validity": 0.9, "average_latency_ms": 100}
    candidate = {"pass_rate": 0.85, "keyword_recall": 0.86, "citation_validity": 0.88, "average_latency_ms": 120}

    comparison = compare_evaluation_summaries(candidate, baseline)

    assert comparison["passed"] is True
    assert comparison["regressions"] == []
    assert comparison["deltas"]["pass_rate"] == -0.05


def test_comparison_reports_quality_and_latency_regressions() -> None:
    baseline = {"pass_rate": 0.9, "keyword_recall": 0.9, "citation_validity": 0.9, "average_latency_ms": 100}
    candidate = {"pass_rate": 0.8, "keyword_recall": 0.9, "citation_validity": 0.9, "average_latency_ms": 140}

    comparison = compare_evaluation_summaries(candidate, baseline)

    assert comparison["passed"] is False
    assert {item["metric"] for item in comparison["regressions"]} == {"pass_rate", "average_latency_ms"}
    assert comparison["deltas"]["latency_increase_ratio"] == 0.4
