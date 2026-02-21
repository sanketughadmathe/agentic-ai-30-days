"""
RAG Error Handling & Retries - Fault-tolerant systems with exponential backoff and circuit breakers
"""

import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -----------------------------
# Pydantic Models
# -----------------------------
class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class ErrorType(str, Enum):
    """Types of errors that can occur."""

    RETRIEVAL_ERROR = "retrieval_error"
    GENERATION_ERROR = "generation_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    CIRCUIT_OPEN = "circuit_open"


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = Field(default=3, description="Maximum retry attempts")
    initial_delay_seconds: float = Field(default=1.0, description="Initial retry delay")
    max_delay_seconds: float = Field(default=60.0, description="Maximum retry delay")
    exponential_base: float = Field(default=2.0, description="Exponential backoff base")
    jitter: bool = Field(default=True, description="Add randomness to delays")


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""

    failure_threshold: int = Field(default=5, description="Failures before opening")
    success_threshold: int = Field(default=2, description="Successes to close")
    timeout_seconds: int = Field(default=60, description="Time before half-open")


class ErrorMetrics(BaseModel):
    """Metrics for a single error."""

    timestamp: str
    error_type: ErrorType
    error_message: str
    retry_attempt: int
    succeeded_after_retry: bool
    total_attempts: int
    total_duration_ms: float


class Answer(BaseModel):
    """RAG answer."""

    answer: str
    confidence: str
    source: str = Field(default="rag", description="Where answer came from")


class ResilientResponse(BaseModel):
    """Response with resilience metadata."""

    status: str  # success, failed, circuit_open
    answer: Answer | None
    error_type: ErrorType | None = None
    error_message: str | None = None
    retries_attempted: int = 0
    total_duration_ms: float = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    fallback_used: bool = False


# -----------------------------
# Exponential Backoff with Jitter
# -----------------------------
class ExponentialBackoff:
    """
    Implements exponential backoff with optional jitter.

    Delays increase exponentially: 1s, 2s, 4s, 8s, ...
    Jitter adds randomness to prevent thundering herd.
    """

    def __init__(self, config: RetryConfig):
        self.config = config

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.

        Formula: min(max_delay, initial_delay * (base ^ attempt))
        With jitter: delay * random(0.5, 1.5)
        """
        delay = self.config.initial_delay_seconds * (
            self.config.exponential_base**attempt
        )
        delay = min(delay, self.config.max_delay_seconds)

        if self.config.jitter:
            import random

            delay *= random.uniform(0.5, 1.5)

        return delay

    def wait(self, attempt: int):
        """Wait for the calculated delay."""
        delay = self.calculate_delay(attempt)
        time.sleep(delay)


# -----------------------------
# Circuit Breaker
# -----------------------------
class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Testing if service recovered

    Prevents cascading failures by failing fast when service is down.
    """

    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.opened_at: datetime | None = None

    def can_execute(self) -> tuple[bool, str | None]:
        """
        Check if request can be executed.

        Returns:
            (can_execute, reason_if_not)
        """
        if self.state == CircuitState.CLOSED:
            return True, None

        if self.state == CircuitState.OPEN:
            # Check if timeout period has passed
            if self.opened_at:
                elapsed = (datetime.now() - self.opened_at).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    # Move to half-open to test recovery
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    return True, None

            return False, f"Circuit breaker '{self.name}' is OPEN"

        # HALF_OPEN: Allow request to test recovery
        return True, None

    def record_success(self):
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                # Service recovered, close circuit
                self._close()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def record_failure(self):
        """Record a failed request."""
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # Still failing, reopen circuit
            self._open()
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self._open()

    def _open(self):
        """Open the circuit."""
        self.state = CircuitState.OPEN
        self.opened_at = datetime.now()
        self.failure_count = 0
        print(f"⚠️  Circuit breaker '{self.name}' OPENED")

    def _close(self):
        """Close the circuit."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.opened_at = None
        print(f"✅ Circuit breaker '{self.name}' CLOSED (recovered)")

    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }


# -----------------------------
# Retry Decorator with Circuit Breaker
# -----------------------------
class ResilientExecutor:
    """
    Executes functions with retries and circuit breaker protection.
    """

    def __init__(
        self,
        retry_config: RetryConfig = RetryConfig(),
        circuit_config: CircuitBreakerConfig = CircuitBreakerConfig(),
    ):
        self.retry_config = retry_config
        self.backoff = ExponentialBackoff(retry_config)
        self.circuit_breaker = CircuitBreaker(circuit_config)

    def execute(
        self, func: Callable, *args, **kwargs
    ) -> tuple[Any, ErrorMetrics | None]:
        """
        Execute function with retries and circuit breaker.

        Returns:
            (result, error_metrics)
        """
        # Check circuit breaker
        can_execute, reason = self.circuit_breaker.can_execute()
        if not can_execute:
            raise Exception(f"Circuit breaker open: {reason}")

        start_time = time.time()
        last_error = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = func(*args, **kwargs)

                # Success!
                self.circuit_breaker.record_success()

                if attempt > 0:
                    # Succeeded after retry
                    duration = (time.time() - start_time) * 1000
                    metrics = ErrorMetrics(
                        timestamp=datetime.now().isoformat(),
                        error_type=ErrorType.RETRIEVAL_ERROR,  # Generic
                        error_message=str(last_error),
                        retry_attempt=attempt,
                        succeeded_after_retry=True,
                        total_attempts=attempt + 1,
                        total_duration_ms=duration,
                    )
                    return result, metrics

                return result, None

            except Exception as e:
                last_error = e

                # Don't retry on last attempt
                if attempt >= self.retry_config.max_retries:
                    self.circuit_breaker.record_failure()
                    break

                # Wait before retry
                print(f"⚠️  Attempt {attempt + 1} failed: {str(e)}")
                print(f"   Retrying in {self.backoff.calculate_delay(attempt):.1f}s...")
                self.backoff.wait(attempt)

        # All retries exhausted
        duration = (time.time() - start_time) * 1000
        metrics = ErrorMetrics(
            timestamp=datetime.now().isoformat(),
            error_type=ErrorType.GENERATION_ERROR,
            error_message=str(last_error),
            retry_attempt=self.retry_config.max_retries,
            succeeded_after_retry=False,
            total_attempts=self.retry_config.max_retries + 1,
            total_duration_ms=duration,
        )

        raise last_error


# -----------------------------
# RAG System with Resilience
# -----------------------------
MODEL = "gemini-2.5-flash"

# Setup RAG components
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

# Separate executors for retrieval and generation
retrieval_executor = ResilientExecutor(
    retry_config=RetryConfig(max_retries=3, initial_delay_seconds=1.0),
    circuit_config=CircuitBreakerConfig(failure_threshold=5, timeout_seconds=30),
)

generation_executor = ResilientExecutor(
    retry_config=RetryConfig(max_retries=2, initial_delay_seconds=2.0),
    circuit_config=CircuitBreakerConfig(failure_threshold=3, timeout_seconds=60),
)


def retrieve_with_retry(question: str, k: int = 3) -> list[Document]:
    """Retrieval with retry logic."""
    return vectorstore.similarity_search(question, k=k)


def generate_with_retry(question: str, docs: list[Document]) -> Answer:
    """Generation with retry logic."""
    llm_structured = llm.with_structured_output(Answer)

    context = "\n".join([f"- {d.page_content}" for d in docs])
    prompt = f"""Answer using only this context:

        {context}

        Question: {question}
    """

    return llm_structured.invoke(prompt)


def resilient_rag(question: str, use_fallback: bool = True) -> ResilientResponse:
    """
    RAG system with comprehensive error handling.

    Features:
    - Exponential backoff retries
    - Circuit breaker protection
    - Graceful fallback
    - Detailed error tracking
    """
    start_time = time.time()
    total_retries = 0

    try:
        # Step 1: Retrieval with retries
        docs, retrieval_metrics = retrieval_executor.execute(
            retrieve_with_retry, question
        )

        if retrieval_metrics:
            total_retries += retrieval_metrics.retry_attempt

        # Step 2: Generation with retries
        answer, generation_metrics = generation_executor.execute(
            generate_with_retry, question, docs
        )

        if generation_metrics:
            total_retries += generation_metrics.retry_attempt

        duration = (time.time() - start_time) * 1000

        return ResilientResponse(
            status="success",
            answer=answer,
            retries_attempted=total_retries,
            total_duration_ms=duration,
            circuit_state=generation_executor.circuit_breaker.state,
            fallback_used=False,
        )

    except Exception as e:
        duration = (time.time() - start_time) * 1000

        # Try fallback
        if use_fallback:
            fallback_answer = Answer(
                answer="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                confidence="LOW",
                source="fallback",
            )

            return ResilientResponse(
                status="failed",
                answer=fallback_answer,
                error_type=ErrorType.GENERATION_ERROR,
                error_message=str(e),
                retries_attempted=total_retries,
                total_duration_ms=duration,
                circuit_state=generation_executor.circuit_breaker.state,
                fallback_used=True,
            )

        # No fallback, return error
        return ResilientResponse(
            status="failed",
            answer=None,
            error_type=ErrorType.GENERATION_ERROR,
            error_message=str(e),
            retries_attempted=total_retries,
            total_duration_ms=duration,
            circuit_state=generation_executor.circuit_breaker.state,
            fallback_used=False,
        )


# -----------------------------
# Pretty Print
# -----------------------------
def print_response(response: ResilientResponse, question: str):
    """Display resilient response."""

    print("\n" + "=" * 70)
    print(f"❓ Question: {question}")
    print("=" * 70)

    # Status
    status_emoji = {"success": "✅", "failed": "❌", "circuit_open": "⚠️"}
    emoji = status_emoji.get(response.status, "❓")
    print(f"\n{emoji} Status: {response.status.upper()}")

    # Answer
    if response.answer:
        print(f"\n💡 Answer: {response.answer.answer}")
        print(f"📊 Confidence: {response.answer.confidence}")
        print(f"📍 Source: {response.answer.source}")

        if response.fallback_used:
            print("⚠️  Using fallback response")

    # Error details
    if response.error_message:
        print(f"\n❌ Error: {response.error_message}")

    # Resilience metrics
    print("\n🔄 Resilience:")
    print(f"   Retries: {response.retries_attempted}")
    print(f"   Duration: {response.total_duration_ms:.1f}ms")
    print(f"   Circuit: {response.circuit_state.value}")

    if response.fallback_used:
        print("   Fallback: Used")

    print("=" * 70)


def print_circuit_status():
    """Display circuit breaker status."""

    print("\n" + "=" * 70)
    print("🔌 CIRCUIT BREAKER STATUS")
    print("=" * 70)

    retrieval_status = retrieval_executor.circuit_breaker.get_status()
    generation_status = generation_executor.circuit_breaker.get_status()

    print("\nRetrieval Circuit:")
    print(f"   State: {retrieval_status['state']}")
    print(f"   Failures: {retrieval_status['failure_count']}")
    print(f"   Successes: {retrieval_status['success_count']}")

    print("\nGeneration Circuit:")
    print(f"   State: {generation_status['state']}")
    print(f"   Failures: {generation_status['failure_count']}")
    print(f"   Successes: {generation_status['success_count']}")

    print("=" * 70)


# -----------------------------
# Testing & Simulation
# -----------------------------
def simulate_transient_failure(question: str):
    """Simulate a transient failure that recovers after retries."""

    print("\n🧪 Simulating transient failure...")

    # Make retrieval fail first 2 times
    original_func = retrieve_with_retry
    call_count = [0]

    def failing_retrieve(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            raise Exception("Simulated transient error")
        return original_func(*args, **kwargs)

    # Temporarily replace function
    globals()["retrieve_with_retry"] = failing_retrieve

    response = resilient_rag(question)

    # Restore original
    globals()["retrieve_with_retry"] = original_func

    return response


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\n🛡️  RAG System with Error Handling & Retries\n")

    # Test 1: Normal operation
    print("=" * 70)
    print("TEST 1: Normal Operation")
    print("=" * 70)

    response1 = resilient_rag("What is ReAct?")
    print_response(response1, "What is ReAct?")

    # Test 2: With fallback
    print("\n" + "=" * 70)
    print("TEST 2: Error with Fallback")
    print("=" * 70)

    # This will succeed, but shows fallback capability
    response2 = resilient_rag("What improves retrieval precision?", use_fallback=True)
    print_response(response2, "What improves retrieval precision?")

    # Test 3: Transient failure (will retry and succeed)
    print("\n" + "=" * 70)
    print("TEST 3: Transient Failure (Retries)")
    print("=" * 70)

    response3 = simulate_transient_failure("How does agent memory work?")
    print_response(response3, "How does agent memory work?")

    # Show circuit breaker status
    print_circuit_status()
