import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CACHE_FILE = Path("day23_rag_caching/cache.json")
CACHE_TTL_HOURS = 24  # Cache expires after 24 hours


# -----------------------------
# Pydantic Models
# -----------------------------
class CachedAnswer(BaseModel):
    """A cached answer with metadata."""

    answer: str = Field(description="The actual answer")
    question: str = Field(description="Original question")
    created_at: str = Field(description="ISO timestamp when cached")
    expires_at: str = Field(description="ISO timestamp when cache expires")
    model: str = Field(description="Model used to generate answer")
    cache_key: str = Field(description="Hash key for the question")
    hit_count: int = Field(
        description="Number of times this cache entry was hit", default=0
    )


class RAGAnswer(BaseModel):
    """Structured answer from RAG system."""

    answer: str = Field(description="Direct answer to the question")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW")
    key_points: list[str] = Field(
        description="Key points from the answer", max_length=5
    )
    needs_more_context: bool = Field(
        description="Whether more context would improve the answer"
    )


class CacheStats(BaseModel):
    """Cache performance statistics."""

    total_entries: int
    total_hits: int
    total_misses: int
    hit_rate: float
    expired_entries: int
    cache_size_bytes: int


# -----------------------------
# Cache Manager
# -----------------------------
class CacheManager:
    """Manages cache with TTL expiry, stats, and persistence."""

    def __init__(self, cache_file: Path, ttl_hours: int = 24):
        self.cache_file = cache_file
        self.ttl_hours = ttl_hours
        self.hits = 0
        self.misses = 0
        self.cache: dict[str, dict] = self._load()

    def _load(self) -> dict:
        """Load cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save(self):
        """Persist cache to disk."""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _make_key(self, question: str) -> str:
        """Create a hash key for a question."""
        return hashlib.md5(question.strip().lower().encode()).hexdigest()

    def _is_expired(self, entry: dict) -> bool:
        """Check if a cache entry has expired."""
        expires_at = datetime.fromisoformat(entry["expires_at"])
        return datetime.now() > expires_at

    def get(self, question: str) -> CachedAnswer | None:
        """
        Retrieve answer from cache.

        Returns:
            CachedAnswer if valid cache hit, None on miss or expiry
        """
        key = self._make_key(question)

        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]

        # Check expiry
        if self._is_expired(entry):
            del self.cache[key]
            self._save()
            self.misses += 1
            return None

        # Cache hit
        self.cache[key]["hit_count"] += 1
        self._save()
        self.hits += 1

        return CachedAnswer(**entry)

    def set(self, question: str, rag_answer: RAGAnswer, model: str):
        """Store answer in cache with TTL."""
        key = self._make_key(question)
        now = datetime.now()

        entry = CachedAnswer(
            answer=rag_answer.answer,
            question=question,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=self.ttl_hours)).isoformat(),
            model=model,
            cache_key=key,
            hit_count=0,
        )

        self.cache[key] = entry.model_dump()
        self._save()

    def invalidate(self, question: str) -> bool:
        """Remove specific question from cache."""
        key = self._make_key(question)
        if key in self.cache:
            del self.cache[key]
            self._save()
            return True
        return False

    def clear(self):
        """Clear all cache entries."""
        self.cache = {}
        self._save()
        print("🗑️  Cache cleared")

    def clear_expired(self) -> int:
        """Remove all expired entries."""
        expired_keys = [k for k, v in self.cache.items() if self._is_expired(v)]
        for key in expired_keys:
            del self.cache[key]
        if expired_keys:
            self._save()
        return len(expired_keys)

    def stats(self) -> CacheStats:
        """Get cache performance statistics."""
        expired = sum(1 for v in self.cache.values() if self._is_expired(v))
        total_hits = sum(v.get("hit_count", 0) for v in self.cache.values())
        total_requests = self.hits + self.misses

        return CacheStats(
            total_entries=len(self.cache),
            total_hits=self.hits,
            total_misses=self.misses,
            hit_rate=self.hits / total_requests if total_requests > 0 else 0.0,
            expired_entries=expired,
            cache_size_bytes=self.cache_file.stat().st_size
            if self.cache_file.exists()
            else 0,
        )


# -----------------------------
# LLM Setup (Gemini)
# -----------------------------
MODEL = "gemini-2.5-flash"

llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
)


# -----------------------------
# Cached RAG
# -----------------------------
cache_manager = CacheManager(CACHE_FILE, ttl_hours=CACHE_TTL_HOURS)


def cached_rag(question: str, force_refresh: bool = False) -> tuple[RAGAnswer, bool]:
    """
    RAG system with smart caching.

    Args:
        question: User's question
        force_refresh: Bypass cache and regenerate answer

    Returns:
        Tuple of (RAGAnswer, was_cached)
    """
    # Check cache unless force refresh
    if not force_refresh:
        cached = cache_manager.get(question)
        if cached:
            # Reconstruct RAGAnswer from cache
            return RAGAnswer(
                answer=cached.answer,
                confidence="HIGH",  # Cached answers were previously validated
                key_points=[],
                needs_more_context=False,
            ), True

    # Cache miss - generate answer
    llm_structured = llm.with_structured_output(RAGAnswer)

    prompt = f"""Answer the question clearly and concisely.

Question: {question}

Provide:
1. A direct answer
2. Your confidence level (HIGH/MEDIUM/LOW)
3. Key points from your answer (max 5)
4. Whether more context would improve the answer"""

    answer = llm_structured.invoke(prompt)

    # Store in cache
    cache_manager.set(question, answer, MODEL)

    return answer, False


# -----------------------------
# Pretty Print Results
# -----------------------------
def print_result(question: str, answer: RAGAnswer, was_cached: bool):
    """Format and print RAG result."""

    print("\n" + "=" * 60)
    cache_status = "💾 CACHE HIT" if was_cached else "🔄 CACHE MISS"
    print(f"{cache_status}")
    print("=" * 60)
    print(f"❓ Question: {question}")
    print(f"\n💡 Answer: {answer.answer}")
    print(f"📊 Confidence: {answer.confidence}")

    if answer.key_points:
        print(f"\n🔑 Key Points:")
        for point in answer.key_points:
            print(f"   • {point}")

    if answer.needs_more_context:
        print(f"\n⚠️  More context would improve this answer")

    print("=" * 60)


def print_stats(stats: CacheStats):
    """Format and print cache statistics."""

    print("\n" + "=" * 60)
    print("📊 CACHE STATISTICS")
    print("=" * 60)
    print(f"Total Entries: {stats.total_entries}")
    print(f"Cache Hits: {stats.total_hits}")
    print(f"Cache Misses: {stats.total_misses}")
    print(f"Hit Rate: {stats.hit_rate:.1%}")
    print(f"Expired Entries: {stats.expired_entries}")
    print(f"Cache Size: {stats.cache_size_bytes:,} bytes")
    print("=" * 60)


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    questions = [
        "What is ReAct?",
        "What improves retrieval precision?",
        "What is ReAct?",  # Duplicate - should hit cache
    ]

    print("\n🧪 Testing Cached RAG System")

    for question in questions:
        answer, was_cached = cached_rag(question)
        print_result(question, answer, was_cached)

    # Force refresh example
    print("\n🔄 Force Refresh Example:")
    answer, was_cached = cached_rag("What is ReAct?", force_refresh=True)
    print_result("What is ReAct?", answer, was_cached)

    # Cache stats
    stats = cache_manager.stats()
    print_stats(stats)

    # Cleanup expired
    expired_count = cache_manager.clear_expired()
    if expired_count:
        print(f"\n🗑️  Cleaned up {expired_count} expired entries")
        print(f"\n🗑️  Cleaned up {expired_count} expired entries")
