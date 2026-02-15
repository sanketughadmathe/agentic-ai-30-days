import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# -----------------------------
# LLM judge
# -----------------------------
judge = ChatOpenAI(
    model="arcee-ai/trinity-large-preview:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# -----------------------------
# Pydantic Models for Structured Evaluation
# -----------------------------
class EvaluationResult(BaseModel):
    """Detailed evaluation of a single answer."""

    score: Literal["CORRECT", "PARTIAL", "INCORRECT"] = Field(
        description="Overall correctness rating"
    )
    factual_accuracy: float = Field(
        description="Factual correctness 0.0-1.0", ge=0.0, le=1.0
    )
    completeness: float = Field(
        description="How complete the answer is 0.0-1.0", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Explanation of the score")
    missing_elements: list[str] = Field(
        description="Key elements from expected answer that are missing", default=[]
    )
    extra_elements: list[str] = Field(
        description="Additional correct information not in expected answer", default=[]
    )


class DatasetMetrics(BaseModel):
    """Aggregate metrics across the entire dataset."""

    total_questions: int
    correct: int
    partial: int
    incorrect: int
    accuracy: float = Field(description="Percentage of CORRECT answers")
    avg_factual_accuracy: float = Field(description="Average factual accuracy score")
    avg_completeness: float = Field(description="Average completeness score")


# -----------------------------
# Sample evaluation dataset
# -----------------------------
dataset = [
    {
        "question": "What is ReAct?",
        "expected": "ReAct combines reasoning and acting in iterative loops.",
    },
    {
        "question": "What improves retrieval precision?",
        "expected": "Reranking improves retrieval precision.",
    },
    {
        "question": "What do structured outputs enforce?",
        "expected": "Structured outputs enforce deterministic contracts.",
    },
]


# -----------------------------
# Simulated RAG answers
# -----------------------------
def rag_system(question: str) -> str:
    """Simulated RAG system - replace with actual implementation."""
    answers = {
        "What is ReAct?": "ReAct combines reasoning and acting.",
        "What improves retrieval precision?": "Reranking improves precision.",
        "What do structured outputs enforce?": "Structured outputs enforce contracts.",
    }
    return answers.get(question, "I don't know.")


# -------------------------------------------
# Judge correctness with structured output
# -------------------------------------------
def evaluate(question: str, expected: str, actual: str) -> EvaluationResult:
    """Evaluate answer quality using LLM judge with detailed breakdown."""

    judge_structured = judge.with_structured_output(EvaluationResult)

    prompt = f"""Evaluate the quality of this answer.

        Question: {question}

        Expected Answer: {expected}

        Actual Answer: {actual}

        Provide:
        1. Overall score (CORRECT/PARTIAL/INCORRECT):
        - CORRECT: Answer is factually accurate and complete
        - PARTIAL: Answer is mostly correct but missing key details or has minor inaccuracies
        - INCORRECT: Answer is wrong or completely misses the point

        2. Factual accuracy (0.0-1.0): How factually correct is the answer?

        3. Completeness (0.0-1.0): How complete is the answer compared to expected?

        4. Reasoning: Brief explanation of your evaluation

        5. Missing elements: What key information from the expected answer is missing?

        6. Extra elements: Any additional correct information not in the expected answer?
    """

    return judge_structured.invoke(prompt)


# -----------------------------
# Calculate aggregate metrics
# -----------------------------
def calculate_metrics(results: list[EvaluationResult]) -> DatasetMetrics:
    """Calculate aggregate metrics from evaluation results."""

    correct = sum(1 for r in results if r.score == "CORRECT")
    partial = sum(1 for r in results if r.score == "PARTIAL")
    incorrect = sum(1 for r in results if r.score == "INCORRECT")

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    avg_factual = sum(r.factual_accuracy for r in results) / total if total > 0 else 0.0
    avg_completeness = (
        sum(r.completeness for r in results) / total if total > 0 else 0.0
    )

    return DatasetMetrics(
        total_questions=total,
        correct=correct,
        partial=partial,
        incorrect=incorrect,
        accuracy=accuracy,
        avg_factual_accuracy=avg_factual,
        avg_completeness=avg_completeness,
    )


# -----------------------------
# Pretty print single result
# -----------------------------
def print_evaluation(
    question: str, expected: str, actual: str, result: EvaluationResult, index: int
):
    """Format and print a single evaluation result."""

    print(f"\n{'=' * 70}")
    print(f"Evaluation {index}")
    print(f"{'=' * 70}")
    print(f"❓ Question: {question}")
    print(f"✓ Expected: {expected}")
    print(f"🤖 Actual: {actual}")
    print(f"\n📊 Score: {result.score}")
    print(f"   Factual Accuracy: {result.factual_accuracy:.2f}")
    print(f"   Completeness: {result.completeness:.2f}")
    print(f"💭 Reasoning: {result.reasoning}")

    if result.missing_elements:
        print("\n❌ Missing Elements:")
        for elem in result.missing_elements:
            print(f"   - {elem}")

    if result.extra_elements:
        print("\n✨ Extra Information:")
        for elem in result.extra_elements:
            print(f"   - {elem}")


# -----------------------------
# Pretty print aggregate metrics
# -----------------------------
def print_metrics(metrics: DatasetMetrics):
    """Format and print aggregate metrics."""

    print(f"\n{'=' * 70}")
    print("OVERALL METRICS")
    print(f"{'=' * 70}")
    print(f"📝 Total Questions: {metrics.total_questions}")
    print(
        f"✅ Correct: {metrics.correct} ({metrics.correct / metrics.total_questions * 100:.1f}%)"
    )
    print(
        f"⚠️  Partial: {metrics.partial} ({metrics.partial / metrics.total_questions * 100:.1f}%)"
    )
    print(
        f"❌ Incorrect: {metrics.incorrect} ({metrics.incorrect / metrics.total_questions * 100:.1f}%)"
    )
    print(f"\n🎯 Accuracy: {metrics.accuracy:.2%}")
    print(f"📈 Avg Factual Accuracy: {metrics.avg_factual_accuracy:.2f}")
    print(f"📊 Avg Completeness: {metrics.avg_completeness:.2f}")
    print(f"{'=' * 70}")


# -----------------------------
# Run evaluation
# -----------------------------
def run_evaluation(dataset: list[dict], rag_fn, verbose: bool = True):
    """
    Run evaluation on dataset.

    Args:
        dataset: List of dicts with 'question' and 'expected' keys
        rag_fn: RAG system function that takes question and returns answer
        verbose: Whether to print individual results
    """
    results = []

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        expected = item["expected"]
        actual = rag_fn(question)

        result = evaluate(question, expected, actual)
        results.append(result)

        if verbose:
            print_evaluation(question, expected, actual, result, i)

    metrics = calculate_metrics(results)
    print_metrics(metrics)

    return results, metrics


# -----------------------------
# Advanced: Compare two RAG systems
# -----------------------------
def compare_systems(
    dataset: list[dict],
    system_a,
    system_b,
    name_a: str = "System A",
    name_b: str = "System B",
):
    """Compare two RAG systems side-by-side."""

    print(f"\n{'=' * 70}")
    print(f"COMPARING: {name_a} vs {name_b}")
    print(f"{'=' * 70}")

    results_a = []
    results_b = []

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        expected = item["expected"]

        actual_a = system_a(question)
        actual_b = system_b(question)

        result_a = evaluate(question, expected, actual_a)
        result_b = evaluate(question, expected, actual_b)

        results_a.append(result_a)
        results_b.append(result_b)

        print(f"\n📝 Question {i}: {question}")
        print(
            f"   {name_a}: {result_a.score} (Factual: {result_a.factual_accuracy:.2f})"
        )
        print(
            f"   {name_b}: {result_b.score} (Factual: {result_b.factual_accuracy:.2f})"
        )

    metrics_a = calculate_metrics(results_a)
    metrics_b = calculate_metrics(results_b)

    print(f"\n{'=' * 70}")
    print(
        f"{name_a}: Accuracy {metrics_a.accuracy:.2%} | Factual {metrics_a.avg_factual_accuracy:.2f}"
    )
    print(
        f"{name_b}: Accuracy {metrics_b.accuracy:.2%} | Factual {metrics_b.avg_factual_accuracy:.2f}"
    )
    print(f"{'=' * 70}")

    return metrics_a, metrics_b


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    print("\n🧪 Running RAG System Evaluation")
    print("=" * 70)

    # Run evaluation
    results, metrics = run_evaluation(dataset, rag_system, verbose=True)

    # Example: Compare with a baseline system
    print("\n\n🔬 Bonus: System Comparison")

    def baseline_system(question: str) -> str:
        """Worse baseline for comparison."""
        return "ReAct is a framework."  # Generic answer for everything

    metrics_rag, metrics_baseline = compare_systems(
        dataset, rag_system, baseline_system, name_a="RAG System", name_b="Baseline"
    )

    # Access structured data programmatically
    print("\n📊 Programmatic Access Example:")
    print(f"First question score: {results[0].score}")
    print(f"First question factual accuracy: {results[0].factual_accuracy}")
    print(f"Overall system accuracy: {metrics.accuracy:.2%}")
    print(f"Overall system accuracy: {metrics.accuracy:.2%}")
