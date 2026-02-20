# Day 25: RAG Rate Limiting & Quotas

## Overview
Production systems need protection from overload. This implementation adds comprehensive rate limiting and quota management to a RAG system with tiered limits, graceful degradation, and fair resource allocation.

## What This Builds
A rate-limited RAG system with:
- **Per-user rate limiting** - Requests per minute/hour/day limits
- **Tiered quotas** - Free/Basic/Pro tiers with different limits
- **Sliding window tracking** - Accurate rate counting over time
- **Graceful degradation** - Clear feedback when limits are hit
- **Quota visibility** - Show users their remaining quota

## Key Concepts

### Why Rate Limiting Matters

**Without rate limiting:**
- One user can exhaust your LLM budget
- Burst traffic can crash your system
- No fair resource allocation
- No cost control

**With rate limiting:**
- Predictable costs
- Fair resource sharing
- System stays responsive under load
- Users know their limits upfront

### Rate Limiting vs Quotas

**Rate Limiting:** How fast you can make requests
- "10 requests per minute"
- Prevents burst abuse
- Short time windows

**Quotas:** Total volume allowed
- "1000 requests per day"
- Manages overall usage
- Longer time windows

Both are needed for production systems.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                User Request                      │
└──────────────────┬───────────────────────────────┘
                   │
                   ↓
         ┌─────────────────────┐
         │   Quota Manager     │
         │  (assigns tier)     │
         └─────────┬───────────┘
                   │
                   ↓
         ┌─────────────────────┐
         │   Rate Limiter      │
         │  (check limits)     │
         └─────────┬───────────┘
                   │
           ┌───────┴───────┐
           │               │
     Under Limit      Over Limit
           │               │
           ↓               ↓
    ┌──────────┐    ┌──────────────┐
    │ Process  │    │ Return 429   │
    │ Request  │    │ + Retry-After│
    └──────────┘    └──────────────┘
```

## Implementation Details

### Sliding Window Rate Limiting

```python
# Track timestamps of all requests
requests = {
    "user_alice": [
        datetime(2024, 2, 18, 10, 30, 15),
        datetime(2024, 2, 18, 10, 30, 25),
        datetime(2024, 2, 18, 10, 30, 35),
    ]
}

# Count requests in last 60 seconds
def count_recent(user_id, window_seconds=60):
    cutoff = datetime.now() - timedelta(seconds=window_seconds)
    return sum(1 for ts in requests[user_id] if ts > cutoff)
```

More accurate than fixed windows, prevents "burst at boundary" exploits.

### Tiered Quotas

```python
tiers = {
    "free": {
        "per_minute": 10,
        "per_hour": 100,
        "per_day": 1000
    },
    "basic": {
        "per_minute": 60,
        "per_hour": 1000,
        "per_day": 10000
    },
    "pro": {
        "per_minute": 1000,
        "per_hour": 100000,
        "per_day": 1000000
    }
}
```

Different users get different limits based on their tier.

### Persistence

Rate limit state is persisted to disk:
```json
{
  "user_alice": [
    "2024-02-18T10:30:15",
    "2024-02-18T10:30:25"
  ],
  "user_bob": [
    "2024-02-18T10:31:05"
  ]
}
```

Survives restarts and enables distributed rate limiting.

## Usage

### Basic Usage
```python
from day25_rag_rate_limiting import rate_limited_rag

# Make a request
response = rate_limited_rag("What is ReAct?", user_id="alice")

if response.status == "success":
    print(f"Answer: {response.answer}")
    print(f"Remaining this minute: {response.quota_status.remaining_minute}")
elif response.status == "rate_limited":
    print(f"Rate limited! Retry in {response.retry_after_seconds}s")
```

### Assign User Tiers
```python
from day25_rag_rate_limiting import quota_manager

# Upgrade a user
quota_manager.set_user_tier("alice", "pro")

# Now alice has higher limits
```

### Check Quota Without Request
```python
# Check user's current quota status
status = quota_manager.check_quota("alice")

print(f"Remaining today: {status.remaining_day}")
print(f"Is rate limited: {status.is_rate_limited}")
```

### Simulate Load
```python
from day25_rag_rate_limiting import simulate_burst

# Test rate limiting with burst traffic
simulate_burst("test_user", num_requests=20, question="Test?")
```

## Sample Output

```
🛡️  RAG System with Rate Limiting

==================================================================
📋 RATE LIMIT TIERS
==================================================================

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
==================================================================

==================================================================
TEST 2: Burst Traffic
==================================================================

🔥 Simulating 15 requests from user 'bob'

✅ Request 1: SUCCESS
✅ Request 2: SUCCESS
✅ Request 3: SUCCESS
...
✅ Request 10: SUCCESS

❌ Request 11: RATE LIMITED
   Reason: requests_per_minute
   Retry after: 45s

📊 Final Status:
   Minute: 10/10
   Hour: 10/100
==================================================================

==================================================================
❓ Question: What is ReAct?
==================================================================

⏱️ Status: RATE_LIMITED

⚠️  Rate limit exceeded (requests_per_minute). Retry in 45s.

📊 Quota Status (User: bob):
   This minute: 10 (remaining: 0)
   This hour: 10 (remaining: 90)
   This day: 10 (remaining: 990)

⏰ Retry After: 45s
   Resets at: 2024-02-18T10:31:00
==================================================================
```

## Rate Limiting Strategies

### 1. Token Bucket (This Implementation)
- Users get tokens at a fixed rate
- Each request consumes a token
- Allows bursts up to bucket size
- **Pros:** Smooth traffic, allows reasonable bursts
- **Cons:** More complex to implement

### 2. Fixed Window
- Count requests in fixed time periods (e.g., per minute starting at :00)
- **Pros:** Simple to implement
- **Cons:** Burst at window boundaries

### 3. Leaky Bucket
- Requests enter a queue, processed at fixed rate
- **Pros:** Perfectly smooth rate
- **Cons:** Can cause delays

### 4. Sliding Log (This Implementation)
- Track timestamp of each request
- Count requests in sliding time window
- **Pros:** Most accurate, no boundary issues
- **Cons:** Requires more memory

## Production Considerations

### Distributed Systems
For multi-server deployments, use Redis:
```python
import redis

class DistributedRateLimiter:
    def __init__(self):
        self.redis = redis.Redis(host='localhost')

    def check_limit(self, user_id):
        key = f"ratelimit:{user_id}:minute"
        count = self.redis.incr(key)

        if count == 1:
            self.redis.expire(key, 60)

        return count <= LIMIT
```

### Cost Optimization
Track cost per request:
```python
# Estimate cost based on token usage
cost_per_request = {
    "retrieval": 0.0001,  # embedding cost
    "generation": 0.002,  # LLM cost
}

# Add to quota tracking
total_cost_today = sum(costs for user_requests)
```

### Monitoring
Alert on:
- High rate limit hit rate (>10% of requests)
- Specific users hitting limits frequently
- Overall quota exhaustion trends

### User Experience
Always include in response:
- Current usage
- Remaining quota
- Reset time
- Clear upgrade path

```json
{
  "status": "rate_limited",
  "quota": {
    "used": 100,
    "limit": 100,
    "resets_at": "2024-02-18T11:00:00Z"
  },
  "message": "Upgrade to Basic for 10x more requests",
  "upgrade_url": "/pricing"
}
```

## Anti-Patterns to Avoid

### ❌ Don't: Silent Rate Limiting
```python
# Bad: Return generic error
if rate_limited:
    return "Error processing request"
```

### ✅ Do: Clear Communication
```python
# Good: Tell user exactly what happened
if rate_limited:
    return {
        "error": "Rate limit exceeded",
        "limit": "10/minute",
        "retry_after": 45,
        "upgrade_url": "/pricing"
    }
```

### ❌ Don't: Client-Side Rate Limiting Only
```python
# Bad: Trust client to rate limit themselves
# Client can bypass this easily
```

### ✅ Do: Server-Side Enforcement
```python
# Good: Enforce limits on server
# Client rate limiting is UX, not security
```

### ❌ Don't: Same Limits for Everyone
```python
# Bad: One size fits all
RATE_LIMIT = 10  # per minute for everyone
```

### ✅ Do: Tiered Quotas
```python
# Good: Different limits based on tier
limits = {
    "free": 10,
    "paid": 100,
    "enterprise": 10000
}
```

## Testing

### Load Testing
```bash
# Generate traffic to test limits
for i in {1..20}; do
  curl -X POST http://localhost:8000/query \
    -H "User-ID: test_user" \
    -d '{"question": "What is RAG?"}'
  sleep 0.1
done
```

### Unit Tests
```python
def test_rate_limiting():
    limiter = RateLimiter(requests_per_minute=5)

    # Should allow first 5 requests
    for i in range(5):
        assert not limiter.check_limit("user1").is_rate_limited
        limiter.record_request("user1")

    # Should block 6th request
    assert limiter.check_limit("user1").is_rate_limited
```

## Next Steps

### Day 26: Error Handling & Retries
Add fault tolerance:
- Exponential backoff
- Circuit breakers
- Automatic retries

### Future Enhancements
1. **Dynamic rate limits** - Adjust based on system load
2. **Priority queuing** - Pro users jump the queue
3. **Cost-based quotas** - Track actual $ spent
4. **Geographic limits** - Different limits per region
5. **Abuse detection** - Auto-ban suspicious patterns

## Files
- `day25_rag_rate_limiting.py` - Main implementation
- `rate_limits_*.json` - Per-tier rate limit state (created on first run)

## Dependencies
```bash
pip install langchain langchain-openai langchain-huggingface \
    langchain-community faiss-cpu python-dotenv pydantic
```

## Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key
TOKENIZERS_PARALLELISM=false
```

## Learn More
- [Token bucket algorithm](https://en.wikipedia.org/wiki/Token_bucket)
- [Rate limiting patterns](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)
- [Redis rate limiting](https://redis.io/docs/manual/patterns/rate-limiter/)
- [Stripe rate limits](https://stripe.com/docs/rate-limits) (excellent example)

---

**Production rule:**
> "Hope is not a strategy for handling traffic spikes."

Rate limiting is insurance for your infrastructure. 🛡️
