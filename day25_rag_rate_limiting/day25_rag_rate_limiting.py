"""
RAG Rate Limiting - Protect production systems from overload
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

RATE_LIMIT_FILE = Path("day25_rag_rate_limiting/rate_limits.json")


# -----------------------------
# Pydantic Models
# -----------------------------
class RateLimitConfig(BaseModel):
    """Rate limit configuration."""

    requests_per_minute: int = Field(description="Max requests per minute", default=10)
    requests_per_hour: int = Field(description="Max requests per hour", default=100)
    requests_per_day: int = Field(description="Max requests per day", default=1000)
    burst_size: int = Field(description="Max burst requests", default=5)


class QuotaStatus(BaseModel):
    """Current quota usage status."""

    user_id: str
    requests_this_minute: int
    requests_this_hour: int
    requests_this_day: int
    remaining_minute: int
    remaining_hour: int
    remaining_day: int
    is_rate_limited: bool
    reset_at: str | None = Field(description="When the limit resets")
    limit_reason: str | None = Field(description="Why rate limited")


class RateLimitedResponse(BaseModel):
    """Response when rate limit is hit."""

    status: Literal["success", "rate_limited", "quota_exceeded"]
    answer: str | None = Field(description="Answer if successful")
    quota_status: QuotaStatus
    retry_after_seconds: int | None = Field(description="Seconds until retry allowed")
    message: str


class Answer(BaseModel):
    """RAG answer."""

    answer: str
    confidence: str


# -----------------------------
# Rate Limiter
# -----------------------------
class RateLimiter:
    """
    Token bucket rate limiter with sliding window.

    Tracks requests per minute/hour/day and enforces limits.
    """

    def __init__(self, config: RateLimitConfig, persist_file: Path = RATE_LIMIT_FILE):
        self.config = config
        self.persist_file = persist_file

        # In-memory tracking: {user_id: {timestamp: request_count}}
        self.requests: dict[str, list[datetime]] = defaultdict(list)

        # Load persisted state
        self._load()

    def _load(self):
        """Load rate limit state from disk."""
        if self.persist_file.exists():
            with open(self.persist_file, "r") as f:
                data = json.load(f)
                for user_id, timestamps_str in data.items():
                    self.requests[user_id] = [
                        datetime.fromisoformat(ts) for ts in timestamps_str
                    ]

    def _save(self):
        """Persist rate limit state to disk."""
        data = {
            user_id: [ts.isoformat() for ts in timestamps]
            for user_id, timestamps in self.requests.items()
        }
        with open(self.persist_file, "w") as f:
            json.dump(data, f, indent=2)

    def _cleanup_old_requests(self, user_id: str):
        """Remove requests older than 24 hours."""
        cutoff = datetime.now() - timedelta(days=1)
        self.requests[user_id] = [ts for ts in self.requests[user_id] if ts > cutoff]

    def _count_requests(self, user_id: str, window_seconds: int) -> int:
        """Count requests within a time window."""
        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        return sum(1 for ts in self.requests[user_id] if ts > cutoff)

    def check_limit(self, user_id: str) -> QuotaStatus:
        """
        Check if user is within rate limits.

        Returns:
            QuotaStatus with current usage and limits
        """
        self._cleanup_old_requests(user_id)

        # Count requests in each window
        req_minute = self._count_requests(user_id, 60)
        req_hour = self._count_requests(user_id, 3600)
        req_day = self._count_requests(user_id, 86400)

        # Check limits
        is_limited = (
            req_minute >= self.config.requests_per_minute
            or req_hour >= self.config.requests_per_hour
            or req_day >= self.config.requests_per_day
        )

        # Determine reset time and reason
        reset_at = None
        limit_reason = None

        if req_minute >= self.config.requests_per_minute:
            reset_at = (datetime.now() + timedelta(seconds=60)).isoformat()
            limit_reason = "requests_per_minute"
        elif req_hour >= self.config.requests_per_hour:
            reset_at = (datetime.now() + timedelta(hours=1)).isoformat()
            limit_reason = "requests_per_hour"
        elif req_day >= self.config.requests_per_day:
            reset_at = (datetime.now() + timedelta(days=1)).isoformat()
            limit_reason = "requests_per_day"

        return QuotaStatus(
            user_id=user_id,
            requests_this_minute=req_minute,
            requests_this_hour=req_hour,
            requests_this_day=req_day,
            remaining_minute=max(0, self.config.requests_per_minute - req_minute),
            remaining_hour=max(0, self.config.requests_per_hour - req_hour),
            remaining_day=max(0, self.config.requests_per_day - req_day),
            is_rate_limited=is_limited,
            reset_at=reset_at,
            limit_reason=limit_reason,
        )

    def record_request(self, user_id: str):
        """Record a request for rate limiting."""
        self.requests[user_id].append(datetime.now())
        self._save()

    def reset_user(self, user_id: str):
        """Reset rate limits for a specific user."""
        if user_id in self.requests:
            del self.requests[user_id]
            self._save()


# -----------------------------
# Quota Manager
# -----------------------------
class QuotaManager:
    """
    Manages user quotas with tiered limits.

    Example tiers:
    - Free: 10/min, 100/hour, 1000/day
    - Basic: 60/min, 1000/hour, 10000/day
    - Pro: Unlimited
    """

    def __init__(self):
        self.tiers = {
            "free": RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=1000,
                burst_size=5,
            ),
            "basic": RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_size=20,
            ),
            "pro": RateLimitConfig(
                requests_per_minute=1000,
                requests_per_hour=100000,
                requests_per_day=1000000,
                burst_size=100,
            ),
        }

        # User tier assignments
        self.user_tiers: dict[str, str] = defaultdict(lambda: "free")

        # Rate limiters per tier
        self.limiters = {
            tier: RateLimiter(
                config, Path(f"day25_rag_rate_limiting/rate_limits_{tier}.json")
            )
            for tier, config in self.tiers.items()
        }

    def set_user_tier(self, user_id: str, tier: str):
        """Assign a user to a tier."""
        if tier not in self.tiers:
            raise ValueError(f"Invalid tier: {tier}")
        self.user_tiers[user_id] = tier

    def check_quota(self, user_id: str) -> QuotaStatus:
        """Check user's quota based on their tier."""
        tier = self.user_tiers[user_id]
        limiter = self.limiters[tier]
        return limiter.check_limit(user_id)

    def record_request(self, user_id: str):
        """Record a request for the user's tier."""
        tier = self.user_tiers[user_id]
        limiter = self.limiters[tier]
        limiter.record_request(user_id)


# -----------------------------
# RAG System with Rate Limiting
# -----------------------------
MODEL = "arcee-ai/trinity-large-preview:free"

# Setup RAG components
documents = [
    Document(page_content="ReAct combines reasoning and acting in iterative loops."),
    Document(page_content="Agent memory stores intermediate reasoning steps."),
    Document(page_content="Structured outputs enforce deterministic contracts."),
    Document(page_content="Reranking improves retrieval precision."),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatOpenAI(
    model=MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)

quota_manager = QuotaManager()


def rate_limited_rag(question: str, user_id: str = "default") -> RateLimitedResponse:
    """
    RAG system with rate limiting and quota management.

    Args:
        question: User's question
        user_id: User identifier for quota tracking

    Returns:
        RateLimitedResponse with answer or rate limit info
    """
    # Check quota
    quota_status = quota_manager.check_quota(user_id)

    if quota_status.is_rate_limited:
        # Calculate retry delay
        if quota_status.reset_at:
            reset_time = datetime.fromisoformat(quota_status.reset_at)
            retry_after = int((reset_time - datetime.now()).total_seconds())
        else:
            retry_after = 60

        return RateLimitedResponse(
            status="rate_limited",
            answer=None,
            quota_status=quota_status,
            retry_after_seconds=retry_after,
            message=f"Rate limit exceeded ({quota_status.limit_reason}). Retry in {retry_after}s.",
        )

    # Record the request
    quota_manager.record_request(user_id)

    # Execute RAG pipeline
    try:
        # Retrieve
        docs = vectorstore.similarity_search(question, k=3)

        # Generate
        llm_structured = llm.with_structured_output(Answer)
        context = "\n".join([f"- {d.page_content}" for d in docs])

        prompt = f"""Answer using only this context:

{context}

Question: {question}"""

        answer_obj = llm_structured.invoke(prompt)

        # Update quota status after successful request
        quota_status = quota_manager.check_quota(user_id)

        return RateLimitedResponse(
            status="success",
            answer=answer_obj.answer,
            quota_status=quota_status,
            retry_after_seconds=None,
            message="Success",
        )

    except Exception as e:
        # Even failed requests count against quota
        quota_status = quota_manager.check_quota(user_id)

        return RateLimitedResponse(
            status="quota_exceeded",
            answer=None,
            quota_status=quota_status,
            retry_after_seconds=None,
            message=f"Error: {str(e)}",
        )


# -----------------------------
# Pretty Print
# -----------------------------
def print_response(response: RateLimitedResponse, question: str):
    """Display rate-limited response."""

    print("\n" + "=" * 70)
    print(f"❓ Question: {question}")
    print("=" * 70)

    # Status
    status_emoji = {"success": "✅", "rate_limited": "⏱️", "quota_exceeded": "❌"}
    emoji = status_emoji.get(response.status, "❓")
    print(f"\n{emoji} Status: {response.status.upper()}")

    # Answer or error
    if response.answer:
        print(f"\n💡 Answer: {response.answer}")
    else:
        print(f"\n⚠️  {response.message}")

    # Quota status
    qs = response.quota_status
    print(f"\n📊 Quota Status (User: {qs.user_id}):")
    print(
        f"   This minute: {qs.requests_this_minute} (remaining: {qs.remaining_minute})"
    )
    print(f"   This hour: {qs.requests_this_hour} (remaining: {qs.remaining_hour})")
    print(f"   This day: {qs.requests_this_day} (remaining: {qs.remaining_day})")

    if response.retry_after_seconds:
        print(f"\n⏰ Retry After: {response.retry_after_seconds}s")
        if qs.reset_at:
            print(f"   Resets at: {qs.reset_at}")

    print("=" * 70)


def print_tier_limits():
    """Display rate limit tiers."""

    print("\n" + "=" * 70)
    print("📋 RATE LIMIT TIERS")
    print("=" * 70)

    for tier_name, config in quota_manager.tiers.items():
        print(f"\n{tier_name.upper()}:")
        print(f"   Per minute: {config.requests_per_minute}")
        print(f"   Per hour: {config.requests_per_hour}")
        print(f"   Per day: {config.requests_per_day}")
        print(f"   Burst size: {config.burst_size}")

    print("=" * 70)


# -----------------------------
# Simulation & Testing
# -----------------------------
def simulate_burst(user_id: str, num_requests: int, question: str):
    """Simulate burst traffic to test rate limiting."""

    print(f"\n🔥 Simulating {num_requests} requests from user '{user_id}'")

    for i in range(num_requests):
        response = rate_limited_rag(question, user_id)

        if response.status == "rate_limited":
            print(f"\n❌ Request {i + 1}: RATE LIMITED")
            print(f"   Reason: {response.quota_status.limit_reason}")
            print(f"   Retry after: {response.retry_after_seconds}s")
            break
        else:
            print(f"✅ Request {i + 1}: SUCCESS")

    # Final quota status
    final_status = quota_manager.check_quota(user_id)
    print("\n📊 Final Status:")
    print(
        f"   Minute: {final_status.requests_this_minute}/{quota_manager.tiers['free'].requests_per_minute}"
    )
    print(
        f"   Hour: {final_status.requests_this_hour}/{quota_manager.tiers['free'].requests_per_hour}"
    )


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\n🛡️  RAG System with Rate Limiting\n")

    # Show tier limits
    print_tier_limits()

    # Test 1: Normal usage
    print("\n" + "=" * 70)
    print("TEST 1: Normal Usage")
    print("=" * 70)

    response1 = rate_limited_rag("What is ReAct?", user_id="alice")
    print_response(response1, "What is ReAct?")

    # Test 2: Burst traffic (should hit rate limit)
    print("\n" + "=" * 70)
    print("TEST 2: Burst Traffic")
    print("=" * 70)

    simulate_burst("bob", 15, "What is ReAct?")

    # Test 3: Different tier
    print("\n" + "=" * 70)
    print("TEST 3: Pro Tier User")
    print("=" * 70)

    quota_manager.set_user_tier("charlie", "pro")
    response3 = rate_limited_rag("What is ReAct?", user_id="charlie")
    print_response(response3, "What is ReAct?")

    # Test 4: Check quota without request
    print("\n" + "=" * 70)
    print("TEST 4: Quota Check (No Request)")
    print("=" * 70)

    status = quota_manager.check_quota("alice")
    print("\nUser 'alice' quota:")
    print(f"   Remaining this minute: {status.remaining_minute}")
    print(f"   Remaining this hour: {status.remaining_hour}")
    print(f"   Remaining this day: {status.remaining_day}")
    print(f"   Is rate limited: {status.is_rate_limited}")

"""output
🛡️  RAG System with Rate Limiting


======================================================================
📋 RATE LIMIT TIERS
======================================================================

FREE:
   Per minute: 10
   Per hour: 100
   Per day: 1000
   Burst size: 5

BASIC:
   Per minute: 60
   Per hour: 1000
   Per day: 10000
   Burst size: 20

PRO:
   Per minute: 1000
   Per hour: 100000
   Per day: 1000000
   Burst size: 100
======================================================================

======================================================================
TEST 1: Normal Usage
======================================================================

======================================================================
❓ Question: What is ReAct?
======================================================================

✅ Status: SUCCESS

💡 Answer: ReAct combines reasoning and acting in iterative loops.

📊 Quota Status (User: alice):
   This minute: 1 (remaining: 9)
   This hour: 1 (remaining: 99)
   This day: 1 (remaining: 999)
======================================================================

======================================================================
TEST 2: Burst Traffic
======================================================================

🔥 Simulating 15 requests from user 'bob'
✅ Request 1: SUCCESS
✅ Request 2: SUCCESS
✅ Request 3: SUCCESS
✅ Request 4: SUCCESS
✅ Request 5: SUCCESS
✅ Request 6: SUCCESS
✅ Request 7: SUCCESS
✅ Request 8: SUCCESS
✅ Request 9: SUCCESS
✅ Request 10: SUCCESS

❌ Request 11: RATE LIMITED
   Reason: requests_per_minute
   Retry after: 59s

📊 Final Status:
   Minute: 10/10
   Hour: 10/100

======================================================================
TEST 3: Pro Tier User
======================================================================

======================================================================
❓ Question: What is ReAct?
======================================================================

✅ Status: SUCCESS

💡 Answer: ReAct combines reasoning and acting in iterative loops.

📊 Quota Status (User: charlie):
   This minute: 1 (remaining: 999)
   This hour: 1 (remaining: 99999)
   This day: 1 (remaining: 999999)
======================================================================

======================================================================
TEST 4: Quota Check (No Request)
======================================================================

User 'alice' quota:
   Remaining this minute: 9
   Remaining this hour: 99
   Remaining this day: 999
   Is rate limited: False
"""
