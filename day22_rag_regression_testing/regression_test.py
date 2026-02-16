# regression_test.py
import json
import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -----------------------------
# Pydantic Models
# -----------------------------
class RegressionAnalysis(BaseModel):
    """Detailed regression analysis for a single question."""

    is_regression: bool = Field(
        description="Whether the actual answer is worse than expected"
    )
    severity: Literal["none", "minor", "major", "critical"] = Field(
        description="Severity of regression if detected"
    )
    quality_score_expected: float = Field(
        description="Quality score for expected answer (0.0-1.0)", ge=0.0, le=1.0
    )
    quality_score_actual: float = Field(
        description="Quality score for actual answer (0.0-1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Explanation of regression assessment")
    specific_issues: list[str] = Field(
        description="Specific problems in the actual answer", default=[]
    )


class RegressionReport(BaseModel):
    """Complete regression test report."""

    timestamp: str
    total_tests: int
    regressions_detected: int
    minor_regressions: int
    major_regressions: int
    critical_regressions: int
    pass_rate: float
    failed_questions: list[str]


# -----------------------------
# Load baseline
# -----------------------------
def load_baseline(filepath: str = "baseline.json") -> dict:
    """Load baseline answers from JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Baseline file '{filepath}' not found. Creating empty baseline.")
        return {}


def save_baseline(baseline: dict, filepath: str = "baseline.json"):
    """Save baseline answers to JSON file."""
    with open(filepath, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"✅ Baseline saved to {filepath}")


# -----------------------------
# Your RAG system
# Replace with your real pipeline
# -----------------------------
def rag_system(question: str) -> str:
    """
    Simulated RAG system - REPLACE WITH YOUR ACTUAL IMPLEMENTATION.

    In production, this would call your full RAG pipeline:
    - retrieve(question)
    - rerank(candidates)
    - generate(context)
    - check_guardrails(answer)
    """
    simulated = {
        "What is ReAct?": "ReAct combines reasoning and acting.",
        "What improves retrieval precision?": "Reranking improves retrieval precision.",
        "What do structured outputs enforce?": "Structured outputs enforce deterministic contracts.",
    }
    return simulated.get(question, "I don't know.")


# -----------------------------
# LLM Judge (Gemini)
# -----------------------------
judge = ChatOpenAI(
    model="arcee-ai/trinity-large-preview:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


def analyze_regression(question: str, expected: str, actual: str) -> RegressionAnalysis:
    """
    Analyze if actual answer represents a regression from expected.

    Args:
        question: The test question
        expected: Previously validated good answer (baseline)
        actual: Current system's answer

    Returns:
        RegressionAnalysis with detailed comparison
    """
    judge_structured = judge.with_structured_output(RegressionAnalysis)

    prompt = f"""Compare these two answers to determine if there's a regression.

        Question: {question}

        Expected Answer (Baseline): {expected}
        Actual Answer (Current): {actual}

        Determine:
        1. Is the actual answer WORSE than the expected answer?
        2. Severity of regression (if any):
        - none: No regression, equal or better quality
        - minor: Slightly less complete but still acceptable
        - major: Significantly worse, missing key information
        - critical: Completely wrong or fails to answer

        3. Quality scores (0.0-1.0) for both answers
        4. Specific issues with the actual answer
        5. Reasoning for your assessment

        A regression occurs when:
        - Factual accuracy decreases
        - Completeness decreases significantly
        - Answer becomes less clear or more confusing
        - Key information is lost

        NOT a regression when:
        - Wording changes but meaning is preserved
        - Answer is equally good but phrased differently
        - Minor stylistic differences
    """

    return judge_structured.invoke(prompt)


# -----------------------------
# Run regression tests
# -----------------------------
def run_regression_tests(
    baseline: dict, rag_fn=rag_system, verbose: bool = True
) -> RegressionReport:
    """
    Run regression tests against baseline.

    Args:
        baseline: Dictionary of {question: expected_answer}
        rag_fn: RAG system function to test
        verbose: Whether to print detailed results

    Returns:
        RegressionReport with summary statistics
    """
    results = []
    regressions = []
    severity_counts = {"minor": 0, "major": 0, "critical": 0}

    if verbose:
        print("\n" + "=" * 70)
        print("REGRESSION TEST SUITE")
        print("=" * 70)

    for question, expected in baseline.items():
        actual = rag_fn(question)
        analysis = analyze_regression(question, expected, actual)
        results.append((question, analysis))

        if verbose:
            print(f"\n📝 Question: {question}")
            print(f"   Expected: {expected}")
            print(f"   Actual: {actual}")
            print(
                f"   Quality: {analysis.quality_score_expected:.2f} → {analysis.quality_score_actual:.2f}"
            )

            if analysis.is_regression:
                emoji = {"minor": "⚠️", "major": "❌", "critical": "🚨"}
                print(
                    f"   {emoji.get(analysis.severity, '❌')} REGRESSION ({analysis.severity.upper()})"
                )
                print(f"   Reasoning: {analysis.reasoning}")
                if analysis.specific_issues:
                    print("   Issues:")
                    for issue in analysis.specific_issues:
                        print(f"     - {issue}")

                regressions.append(question)
                if analysis.severity in severity_counts:
                    severity_counts[analysis.severity] += 1
            else:
                print("   ✅ PASS")

    # Generate report
    total = len(baseline)
    regression_count = len(regressions)
    pass_rate = (total - regression_count) / total if total > 0 else 0.0

    report = RegressionReport(
        timestamp=datetime.now().isoformat(),
        total_tests=total,
        regressions_detected=regression_count,
        minor_regressions=severity_counts["minor"],
        major_regressions=severity_counts["major"],
        critical_regressions=severity_counts["critical"],
        pass_rate=pass_rate,
        failed_questions=regressions,
    )

    if verbose:
        print_report(report)

    return report


# -----------------------------
# Pretty print report
# -----------------------------
def print_report(report: RegressionReport):
    """Print formatted regression test report."""

    print("\n" + "=" * 70)
    print("REGRESSION TEST REPORT")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Total Tests: {report.total_tests}")
    print(f"Pass Rate: {report.pass_rate:.1%}")
    print("\n📊 Results:")
    print(f"   ✅ Passed: {report.total_tests - report.regressions_detected}")
    print(f"   ❌ Failed: {report.regressions_detected}")

    if report.regressions_detected > 0:
        print("\n🔍 Regression Breakdown:")
        print(f"   ⚠️  Minor: {report.minor_regressions}")
        print(f"   ❌ Major: {report.major_regressions}")
        print(f"   🚨 Critical: {report.critical_regressions}")

        print("\n❌ Failed Questions:")
        for q in report.failed_questions:
            print(f"   - {q}")

        # Exit with error code for CI/CD
        print("\n🚨 REGRESSION TESTS FAILED")
    else:
        print("\n✅ ALL REGRESSION TESTS PASSED")

    print("=" * 70)


# -----------------------------
# Save report
# -----------------------------
def save_report(
    report: RegressionReport,
    filepath: str = "regression_report.json",
):
    """Save regression report to JSON file."""
    with open(filepath, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    print(f"📄 Report saved to {filepath}")


# -----------------------------
# Create baseline from current system
# -----------------------------
def create_baseline_from_system(questions: list[str], rag_fn=rag_system) -> dict:
    """
    Create a baseline by running the current system on a set of questions.
    Use this to establish your initial baseline.
    """
    baseline = {}

    print("\n" + "=" * 70)
    print("CREATING BASELINE")
    print("=" * 70)

    for question in questions:
        answer = rag_fn(question)
        baseline[question] = answer
        print(f"\n✓ {question}")
        print(f"  → {answer}")

    return baseline


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Option 1: Run regression tests against existing baseline
    print("\n🧪 Running Regression Tests\n")

    baseline = load_baseline("baseline.json")

    if not baseline:
        # Option 2: Create initial baseline
        print("No baseline found. Creating initial baseline...")
        test_questions = [
            "What is ReAct?",
            "What improves retrieval precision?",
            "What do structured outputs enforce?",
        ]
        baseline = create_baseline_from_system(test_questions, rag_system)
        save_baseline(baseline)
        print("\n✅ Baseline created! Run this script again to test for regressions.")
    else:
        # Run regression tests
        report = run_regression_tests(baseline, rag_system, verbose=True)

        # Save report
        save_report(report)

        # Exit with appropriate code for CI/CD
        exit(0 if report.regressions_detected == 0 else 1)
        exit(0 if report.regressions_detected == 0 else 1)
