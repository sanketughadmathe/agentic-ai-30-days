# day27_multi_provider_resilience.py
"""
Multi-Provider Resilience - Failover between LLM providers with intelligent degradation
"""

import os
import time
from datetime import datetime, timedelta
from enum import Enum

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
class ProviderStatus(str, Enum):
    """Provider health states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    COOLDOWN = "cooldown"
    UNAVAILABLE = "unavailable"


class ModelTier(str, Enum):
    """Quality/cost tiers for models."""

    PREMIUM = "premium"  # GPT-4, Claude Opus
    STANDARD = "standard"  # GPT-3.5, Claude Sonnet
    BUDGET = "budget"  # Local models, smaller models
    FALLBACK = "fallback"  # Cached/static responses


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    name: str = Field(description="Provider identifier")
    model: str = Field(description="Model name")
    tier: ModelTier = Field(description="Quality/cost tier")
    api_key_env: str = Field(description="Environment variable for API key")
    base_url: str | None = Field(default=None, description="Custom base URL")
    cost_per_1k_tokens: float = Field(description="Approximate cost")
    max_retries: int = Field(default=2, description="Retries before cooldown")
    cooldown_minutes: int = Field(default=5, description="Cooldown duration")


class ProviderHealth(BaseModel):
    """Health metrics for a provider."""

    name: str
    status: ProviderStatus
    consecutive_failures: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    last_failure_time: str | None
    cooldown_until: str | None
    success_rate: float
    avg_latency_ms: float


class Answer(BaseModel):
    """Generated answer."""

    answer: str
    confidence: str
    source: str = Field(description="Which model generated this")


class MultiProviderResponse(BaseModel):
    """Response with provider metadata."""

    status: str  # success, partial_success, failed
    answer: Answer | None
    provider_used: str | None
    tier_used: ModelTier | None
    failovers_attempted: int
    providers_tried: list[str]
    total_latency_ms: float
    error_message: str | None = None


# -----------------------------
# Provider Manager
# -----------------------------
class ProviderManager:
    """
    Manages multiple LLM providers with health tracking and failover.

    Features:
    - Automatic failover on errors
    - Provider cooldowns after repeated failures
    - Tier-based degradation (premium → standard → budget)
    - Health tracking and metrics
    """

    def __init__(self, providers: list[ProviderConfig]):
        self.providers = {p.name: p for p in providers}

        # Health tracking
        self.health: dict[str, ProviderHealth] = {
            name: ProviderHealth(
                name=name,
                status=ProviderStatus.HEALTHY,
                consecutive_failures=0,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                last_failure_time=None,
                cooldown_until=None,
                success_rate=1.0,
                avg_latency_ms=0.0,
            )
            for name in self.providers
        }

        # Latency tracking
        self.latencies: dict[str, list[float]] = {name: [] for name in self.providers}

    def get_available_providers(self, tier: ModelTier | None = None) -> list[str]:
        """
        Get providers that are available (not in cooldown).

        Args:
            tier: Optional tier filter

        Returns:
            List of available provider names, sorted by tier then success rate
        """
        available = []

        for name, config in self.providers.items():
            # Check tier filter
            if tier and config.tier != tier:
                continue

            # Check cooldown
            health = self.health[name]
            if health.status == ProviderStatus.COOLDOWN:
                # Check if cooldown expired
                if health.cooldown_until:
                    cooldown_time = datetime.fromisoformat(health.cooldown_until)
                    if datetime.now() >= cooldown_time:
                        # Cooldown expired, mark as healthy
                        self._recover_from_cooldown(name)
                        available.append(name)
                    # else: still in cooldown, skip
                # else: skip (in cooldown)
            elif health.status != ProviderStatus.UNAVAILABLE:
                available.append(name)

        # Sort by tier (premium first) then success rate
        tier_order = {
            ModelTier.PREMIUM: 0,
            ModelTier.STANDARD: 1,
            ModelTier.BUDGET: 2,
            ModelTier.FALLBACK: 3,
        }
        available.sort(
            key=lambda n: (
                tier_order[self.providers[n].tier],
                -self.health[n].success_rate,  # Higher success rate first
            )
        )

        return available

    def record_success(self, provider_name: str, latency_ms: float):
        """Record successful request."""
        health = self.health[provider_name]

        health.total_requests += 1
        health.successful_requests += 1
        health.consecutive_failures = 0
        health.success_rate = health.successful_requests / health.total_requests

        # Track latency
        self.latencies[provider_name].append(latency_ms)
        # Keep only last 100 latencies
        if len(self.latencies[provider_name]) > 100:
            self.latencies[provider_name] = self.latencies[provider_name][-100:]

        health.avg_latency_ms = sum(self.latencies[provider_name]) / len(
            self.latencies[provider_name]
        )

        # If was degraded, might recover
        if health.status == ProviderStatus.DEGRADED:
            if health.consecutive_failures == 0:
                health.status = ProviderStatus.HEALTHY

    def record_failure(self, provider_name: str, error: Exception):
        """Record failed request and potentially put in cooldown."""
        health = self.health[provider_name]
        config = self.providers[provider_name]

        health.total_requests += 1
        health.failed_requests += 1
        health.consecutive_failures += 1
        health.last_failure_time = datetime.now().isoformat()
        health.success_rate = health.successful_requests / health.total_requests

        # Check if should enter cooldown
        if health.consecutive_failures >= config.max_retries:
            self._enter_cooldown(provider_name, config.cooldown_minutes)
        elif health.consecutive_failures >= 1:
            health.status = ProviderStatus.DEGRADED

    def _enter_cooldown(self, provider_name: str, minutes: int):
        """Put provider in cooldown."""
        health = self.health[provider_name]
        health.status = ProviderStatus.COOLDOWN
        health.cooldown_until = (
            datetime.now() + timedelta(minutes=minutes)
        ).isoformat()

        print(
            f"⏸️  Provider '{provider_name}' entering {minutes}min cooldown after {health.consecutive_failures} failures"
        )

    def _recover_from_cooldown(self, provider_name: str):
        """Recover provider from cooldown."""
        health = self.health[provider_name]
        health.status = ProviderStatus.DEGRADED  # Start as degraded, not healthy
        health.cooldown_until = None
        health.consecutive_failures = 0

        print(
            f"🔄 Provider '{provider_name}' recovered from cooldown (marked as degraded)"
        )

    def get_health_report(self) -> dict[str, dict]:
        """Get comprehensive health report."""
        return {
            name: {
                "status": health.status.value,
                "tier": self.providers[name].tier.value,
                "success_rate": f"{health.success_rate:.1%}",
                "avg_latency_ms": f"{health.avg_latency_ms:.1f}ms",
                "consecutive_failures": health.consecutive_failures,
                "total_requests": health.total_requests,
                "cooldown_until": health.cooldown_until,
            }
            for name, health in self.health.items()
        }


# -----------------------------
# Provider Configurations
# -----------------------------
PROVIDERS = [
    # Premium tier - Gemini
    ProviderConfig(
        name="gemini-flash",
        model="gemini-3-flash-preview",
        tier=ModelTier.PREMIUM,
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        cost_per_1k_tokens=0.002,
        max_retries=2,
        cooldown_minutes=5,
    ),
    # Standard tier - OpenRouter Arcee Trinity (free)
    ProviderConfig(
        name="arcee-trinity",
        model="arcee-ai/trinity-large-preview:free",
        tier=ModelTier.STANDARD,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        cost_per_1k_tokens=0.0,  # Free tier
        max_retries=3,
        cooldown_minutes=3,
    ),
    # Note: This demonstrates multi-provider failover with:
    # - Gemini (premium, paid)
    # - OpenRouter Arcee Trinity (standard, free)
    # In production, add more providers for redundancy
]

provider_manager = ProviderManager(PROVIDERS)


# -----------------------------
# RAG Setup
# -----------------------------
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


def get_llm_client(provider_name: str) -> ChatOpenAI:
    """Create LLM client for a provider."""
    config = provider_manager.providers[provider_name]

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise ValueError(f"API key not found: {config.api_key_env}")

    kwargs = {"model": config.model, "api_key": api_key, "temperature": 0}

    if config.base_url:
        kwargs["base_url"] = config.base_url

    return ChatOpenAI(**kwargs)


# -----------------------------
# Multi-Provider RAG
# -----------------------------
def multi_provider_rag(
    question: str, preferred_tier: ModelTier | None = None
) -> MultiProviderResponse:
    """
    RAG with automatic provider failover.

    Strategy:
    1. Try premium tier first (if available)
    2. On failure, failover to next tier
    3. Track provider health and apply cooldowns
    4. Return best available answer

    Args:
        question: User's question
        preferred_tier: Optional tier preference

    Returns:
        MultiProviderResponse with answer and metadata
    """
    start_time = time.time()
    providers_tried = []
    failovers = 0

    # Get retrieval context (shared across all providers)
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([f"- {d.page_content}" for d in docs])

    # Get available providers, optionally filtered by tier
    available = provider_manager.get_available_providers(tier=preferred_tier)

    if not available:
        return MultiProviderResponse(
            status="failed",
            answer=None,
            provider_used=None,
            tier_used=None,
            failovers_attempted=0,
            providers_tried=[],
            total_latency_ms=(time.time() - start_time) * 1000,
            error_message="No providers available",
        )

    # Try each provider until one succeeds
    for provider_name in available:
        providers_tried.append(provider_name)
        config = provider_manager.providers[provider_name]

        print(f"\n🔄 Trying provider: {provider_name} (tier: {config.tier.value})")

        try:
            # Create LLM client
            llm = get_llm_client(provider_name)
            llm_structured = llm.with_structured_output(Answer)

            # Generate answer
            provider_start = time.time()

            prompt = f"""Answer using only this context:

                {context}

                Question: {question}

                Provide your answer with confidence level.
            """

            answer = llm_structured.invoke(prompt)
            answer.source = f"{provider_name} ({config.tier.value})"

            provider_latency = (time.time() - provider_start) * 1000

            # Record success
            provider_manager.record_success(provider_name, provider_latency)

            total_latency = (time.time() - start_time) * 1000

            return MultiProviderResponse(
                status="success",
                answer=answer,
                provider_used=provider_name,
                tier_used=config.tier,
                failovers_attempted=failovers,
                providers_tried=providers_tried,
                total_latency_ms=total_latency,
            )

        except Exception as e:
            print(f"❌ Provider '{provider_name}' failed: {str(e)}")

            # Record failure
            provider_manager.record_failure(provider_name, e)

            # Check for rate limit error (special handling)
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(
                    f"⚠️  Rate limit hit on '{provider_name}' - treating as failover signal"
                )

            failovers += 1

            # Try next provider
            continue

    # All providers failed
    total_latency = (time.time() - start_time) * 1000

    return MultiProviderResponse(
        status="failed",
        answer=None,
        provider_used=None,
        tier_used=None,
        failovers_attempted=failovers,
        providers_tried=providers_tried,
        total_latency_ms=total_latency,
        error_message=f"All {len(providers_tried)} providers failed",
    )


# -----------------------------
# Pretty Print
# -----------------------------
def print_response(response: MultiProviderResponse, question: str):
    """Display multi-provider response."""

    print("\n" + "=" * 70)
    print(f"❓ Question: {question}")
    print("=" * 70)

    # Status
    status_emoji = {"success": "✅", "failed": "❌"}
    emoji = status_emoji.get(response.status, "❓")
    print(f"\n{emoji} Status: {response.status.upper()}")

    # Answer
    if response.answer:
        print(f"\n💡 Answer: {response.answer.answer}")
        print(f"📊 Confidence: {response.answer.confidence}")
        print(f"🤖 Source: {response.answer.source}")

    # Provider info
    print("\n🔄 Provider Journey:")
    print(f"   Used: {response.provider_used or 'None'}")
    if response.tier_used:
        print(f"   Tier: {response.tier_used.value}")
    print(f"   Tried: {', '.join(response.providers_tried)}")
    print(f"   Failovers: {response.failovers_attempted}")
    print(f"   Latency: {response.total_latency_ms:.1f}ms")

    if response.error_message:
        print(f"\n❌ Error: {response.error_message}")

    print("=" * 70)


def print_health_dashboard():
    """Display provider health dashboard."""

    health_report = provider_manager.get_health_report()

    print("\n" + "=" * 70)
    print("🏥 PROVIDER HEALTH DASHBOARD")
    print("=" * 70)

    for name, metrics in health_report.items():
        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "cooldown": "⏸️",
            "unavailable": "❌",
        }
        emoji = status_emoji.get(metrics["status"], "❓")

        print(f"\n{emoji} {name.upper()} ({metrics['tier']})")
        print(f"   Status: {metrics['status']}")
        print(f"   Success Rate: {metrics['success_rate']}")
        print(f"   Avg Latency: {metrics['avg_latency_ms']}")
        print(f"   Total Requests: {metrics['total_requests']}")

        if metrics["consecutive_failures"] > 0:
            print(f"   Consecutive Failures: {metrics['consecutive_failures']}")

        if metrics["cooldown_until"]:
            print(f"   Cooldown Until: {metrics['cooldown_until']}")

    print("=" * 70)


# -----------------------------
# Testing & Simulation
# -----------------------------
def simulate_provider_failure(provider_name: str, num_failures: int = 3):
    """Simulate provider failures to test cooldown."""

    print(f"\n🧪 Simulating {num_failures} failures on '{provider_name}'")

    for i in range(num_failures):
        provider_manager.record_failure(
            provider_name, Exception(f"Simulated failure {i + 1}")
        )
        print(f"   Failure {i + 1} recorded")

    health = provider_manager.health[provider_name]
    print(f"\n   Status: {health.status.value}")
    if health.cooldown_until:
        print(f"   Cooldown until: {health.cooldown_until}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\n🌐 Multi-Provider RAG with Intelligent Failover\n")

    # Test 1: Normal operation
    print("=" * 70)
    print("TEST 1: Normal Operation")
    print("=" * 70)

    response1 = multi_provider_rag("What is ReAct?")
    print_response(response1, "What is ReAct?")

    # Test 2: Simulate provider failure and watch failover
    print("\n" + "=" * 70)
    print("TEST 2: Provider Failure → Automatic Failover")
    print("=" * 70)

    # Put primary provider in cooldown
    simulate_provider_failure("gemini-flash", num_failures=2)

    # This should automatically failover to secondary
    response2 = multi_provider_rag("What improves retrieval precision?")
    print_response(response2, "What improves retrieval precision?")

    # Test 3: Multiple requests showing load distribution
    print("\n" + "=" * 70)
    print("TEST 3: Load Distribution")
    print("=" * 70)

    questions = [
        "How does agent memory work?",
        "What are structured outputs?",
    ]

    for q in questions:
        response = multi_provider_rag(q)
        print(f"\n✓ '{q}' → {response.provider_used}")

    # Show final health dashboard
    print_health_dashboard()

    print("\n" + "=" * 70)
    print("💡 Key Insights:")
    print("=" * 70)
    print("• Rate limits trigger automatic failover (not just retries)")
    print("• Failed providers enter cooldown to prevent retry storms")
    print("• System degrades gracefully through tiers")
    print("• Health tracking enables smart routing decisions")
    print("=" * 70)
