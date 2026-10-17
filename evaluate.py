"""Small reproducible retrieval evaluation using a JSONL test set."""
import json
from pathlib import Path
from retrieve_generate import answer_question

BASE_DIR = Path(__file__).resolve().parent
EVAL_FILE = BASE_DIR / "evaluation" / "questions.jsonl"
REPORT_FILE = BASE_DIR / "evaluation" / "results.json"


def main() -> None:
    if not EVAL_FILE.exists():
        raise SystemExit(f"Evaluation file not found: {EVAL_FILE}")

    cases = []
    with EVAL_FILE.open("r", encoding="utf-8") as file:
        cases = [json.loads(line) for line in file if line.strip()]

    if not cases:
        raise SystemExit("Add at least one evaluation case to questions.jsonl.")

    rows = []
    retrieval_hits = 0
    answer_contains_expected = 0

    for case in cases:
        result = answer_question(case["question"], k=4)
        retrieved_sources = {s["source"].replace("\\", "/") for s in result["sources"]}
        expected_source = case.get("expected_source", "").replace("\\", "/")
        hit = bool(expected_source and any(expected_source in src for src in retrieved_sources))
        retrieval_hits += int(hit)

        expected_terms = case.get("expected_terms", [])
        normalized_answer = result["answer"].lower()
        term_hit = bool(expected_terms) and all(term.lower() in normalized_answer for term in expected_terms)
        answer_contains_expected += int(term_hit)

        rows.append({
            "question": case["question"],
            "expected_source": expected_source,
            "retrieval_hit": hit,
            "expected_terms_present": term_hit,
            "answer": result["answer"],
            "retrieved_sources": list(retrieved_sources),
        })
        print(f"Completed: {case['question']}")

    report = {
        "number_of_questions": len(cases),
        "source_hit_rate_at_4": retrieval_hits / len(cases),
        "expected_term_coverage": answer_contains_expected / len(cases),
        "note": (
            "These are lightweight proxy metrics, not a complete semantic evaluation. "
            "Manually review correctness, citation support, and abstention on every answer."
        ),
        "results": rows,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nEvaluation summary")
    print(f"Questions: {len(cases)}")
    print(f"Source hit rate @4: {report['source_hit_rate_at_4']:.1%}")
    print(f"Expected-term coverage: {report['expected_term_coverage']:.1%}")
    print(f"Detailed report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
