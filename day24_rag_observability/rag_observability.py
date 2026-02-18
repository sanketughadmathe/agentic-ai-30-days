"""
RAG Observability - Track retrieval quality, latency, and system health
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

METRICS_FILE = Path("day24_rag_observability/rag_metrics.jsonl")


# -----------------------------
# Pydantic Models
# -----------------------------
class RetrievalMetrics(BaseModel):
    """Metrics for a single retrieval operation."""

    timestamp: str = Field(description="ISO timestamp")
    question: str = Field(description="User's question")
    retrieval_latency_ms: float = Field(description="Time to retrieve documents")
    num_candidates: int = Field(description="Number of documents retrieved")
    rerank_latency_ms: float | None = Field(description="Time to rerank", default=None)
    generation_latency_ms: float = Field(description="Time to generate answer")
    total_latency_ms: float = Field(description="End-to-end latency")
    cache_hit: bool = Field(description="Whether answer came from cache")

    # Quality metrics
    answer_length: int = Field(description="Length of generated answer")
    confidence: str | None = Field(description="Answer confidence level", default=None)
    grounded: bool | None = Field(
        description="Whether answer is grounded", default=None
    )

    # System metrics
    model_used: str = Field(description="LLM model name")
    error: str | None = Field(description="Error message if failed", default=None)


class SystemHealth(BaseModel):
    """Aggregate system health metrics."""

    time_window_hours: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    cache_hit_rate: float

    avg_retrieval_latency_ms: float
    p95_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    p95_generation_latency_ms: float
    avg_total_latency_ms: float
    p95_total_latency_ms: float

    success_rate: float


class Answer(BaseModel):
    """Structured answer from RAG."""

    answer: str
    confidence: str = Field(description="HIGH, MEDIUM, or LOW")
    grounded: bool = Field(description="Answer is grounded in context")


# -----------------------------
# Observability Layer
# -----------------------------
class RAGObserver:
    """Tracks and persists RAG system metrics."""

    def __init__(self, metrics_file: Path):
        self.metrics_file = metrics_file
        self.current_metrics: dict[str, Any] = {}

    def start_span(self, span_name: str):
        """Start timing a span."""
        self.current_metrics[f"{span_name}_start"] = time.time()

    def end_span(self, span_name: str) -> float:
        """End timing a span and return duration in ms."""
        start_key = f"{span_name}_start"
        if start_key not in self.current_metrics:
            return 0.0

        duration_ms = (time.time() - self.current_metrics[start_key]) * 1000
        del self.current_metrics[start_key]
        return duration_ms

    def log_metrics(self, metrics: RetrievalMetrics):
        """Persist metrics to JSONL file."""
        with open(self.metrics_file, "a") as f:
            f.write(metrics.model_dump_json() + "\n")

    def load_metrics(self, time_window_hours: float = 24) -> list[RetrievalMetrics]:
        """Load recent metrics from file."""
        if not self.metrics_file.exists():
            return []

        cutoff = datetime.now().timestamp() - (time_window_hours * 3600)
        metrics = []

        with open(self.metrics_file, "r") as f:
            for line in f:
                m = RetrievalMetrics.model_validate_json(line)
                ts = datetime.fromisoformat(m.timestamp).timestamp()
                if ts >= cutoff:
                    metrics.append(m)

        return metrics

    def compute_health(self, time_window_hours: float = 24) -> SystemHealth:
        """Compute aggregate system health metrics."""
        metrics = self.load_metrics(time_window_hours)

        if not metrics:
            return SystemHealth(
                time_window_hours=time_window_hours,
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                cache_hit_rate=0.0,
                avg_retrieval_latency_ms=0.0,
                p95_retrieval_latency_ms=0.0,
                avg_generation_latency_ms=0.0,
                p95_generation_latency_ms=0.0,
                avg_total_latency_ms=0.0,
                p95_total_latency_ms=0.0,
                success_rate=0.0,
            )

        total = len(metrics)
        successful = [m for m in metrics if m.error is None]
        failed = [m for m in metrics if m.error is not None]
        cache_hits = [m for m in metrics if m.cache_hit]

        retrieval_latencies = [m.retrieval_latency_ms for m in successful]
        generation_latencies = [m.generation_latency_ms for m in successful]
        total_latencies = [m.total_latency_ms for m in successful]

        def percentile(data: list[float], p: int) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            index = int(len(sorted_data) * p / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]

        return SystemHealth(
            time_window_hours=time_window_hours,
            total_queries=total,
            successful_queries=len(successful),
            failed_queries=len(failed),
            cache_hit_rate=len(cache_hits) / total if total > 0 else 0.0,
            avg_retrieval_latency_ms=sum(retrieval_latencies) / len(retrieval_latencies)
            if retrieval_latencies
            else 0.0,
            p95_retrieval_latency_ms=percentile(retrieval_latencies, 95),
            avg_generation_latency_ms=sum(generation_latencies)
            / len(generation_latencies)
            if generation_latencies
            else 0.0,
            p95_generation_latency_ms=percentile(generation_latencies, 95),
            avg_total_latency_ms=sum(total_latencies) / len(total_latencies)
            if total_latencies
            else 0.0,
            p95_total_latency_ms=percentile(total_latencies, 95),
            success_rate=len(successful) / total if total > 0 else 0.0,
        )


# -----------------------------
# RAG System with Observability
# -----------------------------
MODEL = "gemini-2.5-flash"

# Setup
documents = [
    Document(page_content="ReAct combines reasoning and acting in iterative loops."),
    Document(page_content="Agent memory stores intermediate reasoning steps."),
    Document(page_content="Structured outputs enforce deterministic contracts."),
    Document(page_content="Reranking improves retrieval precision."),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
)

observer = RAGObserver(METRICS_FILE)


def observed_rag(
    question: str, cache_hit: bool = False
) -> tuple[Answer, RetrievalMetrics]:
    """
    RAG system with full observability.

    Returns:
        Tuple of (Answer, RetrievalMetrics)
    """
    observer.start_span("total")
    error = None
    answer_obj = None

    try:
        # Retrieval
        observer.start_span("retrieval")
        docs = vectorstore.similarity_search(question, k=3)
        retrieval_latency = observer.end_span("retrieval")

        # Generation
        observer.start_span("generation")
        llm_structured = llm.with_structured_output(Answer)

        context = "\n".join([f"- {d.page_content}" for d in docs])
        prompt = f"""Answer using only this context:

            {context}

            Question: {question}

            Provide answer, confidence (HIGH/MEDIUM/LOW), and whether it's grounded in the context.
        """

        answer_obj = llm_structured.invoke(prompt)
        generation_latency = observer.end_span("generation")

    except Exception as e:
        error = str(e)
        retrieval_latency = observer.end_span("retrieval")
        generation_latency = observer.end_span("generation")
        answer_obj = Answer(answer="Error occurred", confidence="LOW", grounded=False)

    total_latency = observer.end_span("total")

    # Create metrics
    metrics = RetrievalMetrics(
        timestamp=datetime.now().isoformat(),
        question=question,
        retrieval_latency_ms=retrieval_latency,
        num_candidates=len(docs) if not error else 0,
        generation_latency_ms=generation_latency,
        total_latency_ms=total_latency,
        cache_hit=cache_hit,
        answer_length=len(answer_obj.answer),
        confidence=answer_obj.confidence,
        grounded=answer_obj.grounded,
        model_used=MODEL,
        error=error,
    )

    observer.log_metrics(metrics)

    return answer_obj, metrics


# -----------------------------
# Pretty Print
# -----------------------------
def print_answer_with_metrics(question: str, answer: Answer, metrics: RetrievalMetrics):
    """Display answer with observability data."""

    print("\n" + "=" * 70)
    print(f"❓ Question: {question}")
    print("=" * 70)
    print(f"💡 Answer: {answer.answer}")
    print(f"📊 Confidence: {answer.confidence}")
    print(f"✓ Grounded: {'Yes' if answer.grounded else 'No'}")

    print("\n⏱️  Latency Breakdown:")
    print(f"   Retrieval: {metrics.retrieval_latency_ms:.1f}ms")
    print(f"   Generation: {metrics.generation_latency_ms:.1f}ms")
    print(f"   Total: {metrics.total_latency_ms:.1f}ms")

    print("\n📈 Metadata:")
    print(f"   Candidates: {metrics.num_candidates}")
    print(f"   Answer Length: {metrics.answer_length}")
    print(f"   Cache Hit: {metrics.cache_hit}")
    print(f"   Model: {metrics.model_used}")

    if metrics.error:
        print(f"\n❌ Error: {metrics.error}")

    print("=" * 70)


def print_health(health: SystemHealth):
    """Display system health dashboard."""

    print("\n" + "=" * 70)
    print(f"📊 SYSTEM HEALTH (Last {health.time_window_hours}h)")
    print("=" * 70)

    print("\n📝 Volume:")
    print(f"   Total Queries: {health.total_queries}")
    print(f"   ✅ Successful: {health.successful_queries}")
    print(f"   ❌ Failed: {health.failed_queries}")
    print(f"   Success Rate: {health.success_rate:.1%}")

    print("\n💾 Caching:")
    print(f"   Cache Hit Rate: {health.cache_hit_rate:.1%}")

    print("\n⏱️  Latency (ms):")
    print("   Retrieval:")
    print(f"      Avg: {health.avg_retrieval_latency_ms:.1f}ms")
    print(f"      P95: {health.p95_retrieval_latency_ms:.1f}ms")
    print("   Generation:")
    print(f"      Avg: {health.avg_generation_latency_ms:.1f}ms")
    print(f"      P95: {health.p95_generation_latency_ms:.1f}ms")
    print("   Total:")
    print(f"      Avg: {health.avg_total_latency_ms:.1f}ms")
    print(f"      P95: {health.p95_total_latency_ms:.1f}ms")

    print("=" * 70)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\n🔍 RAG System with Observability")

    # Run some queries
    questions = [
        "What is ReAct?",
        "What improves retrieval precision?",
        "What is ReAct?",  # Simulated cache hit
        "How does agent memory work?",
    ]

    for i, question in enumerate(questions):
        # Simulate cache hit on duplicate
        cache_hit = i == 2

        answer, metrics = observed_rag(question, cache_hit=cache_hit)
        print_answer_with_metrics(question, answer, metrics)

    # Show system health
    health = observer.compute_health(time_window_hours=24)
    print_health(health)
    print_health(health)
