# Day 26: RAG Error Handling & Retries

## Overview
Production systems must handle failures gracefully. This implementation adds comprehensive error handling with exponential backoff retries, circuit breakers, and intelligent fallback strategies.

## What This Builds
A fault-tolerant RAG system with:
- **Exponential backoff retries** - Smart retry logic with increasing delays
- **Circuit breakers** - Fail fast when service is down
- **Graceful fallback** - Provide alternative responses when primary fails
- **Error tracking** - Monitor and measure resilience
- **Jitter** - Prevent thundering herd problem

## Key Concepts

### Why Error Handling Matters

**Without resilience:**
- Transient errors cause complete failures
- Cascading failures take down entire system
- Users see cryptic error messages
- No visibility into retry behavior

**With resilience:**
- Transient errors auto-recover
- Circuit breakers prevent cascades
- Users get graceful fallbacks
- Clear metrics on failure patterns

### The Three Pillars of Resilience

1. **Retries** - Try again when failures are transient
2. **Circuit Breakers** - Stop trying when failures are persistent
3. **Fallbacks** - Provide alternative when primary fails

## Architecture

```
┌──────────────────────────────────────────────────┐
│                User Request                      │
└──────────────────┬───────────────────────────────┘
                   │
                   ↓
         ┌─────────────────────┐
         │  Circuit Breaker    │
         │  (Can execute?)     │
         └─────────┬───────────┘
                   │
           ┌───────┴────────┐
           │                │
        CLOSED           OPEN
           │                │
           ↓                ↓
    ┌──────────┐    ┌─────────────┐
    │ Execute  │    │ Fail Fast   │
    │ Function │    │ (no retry)  │
    └────┬─────┘    └─────────────┘
         │
     ┌───┴───┐
  Success  Failure
     │       │
     ↓       ↓
  Return  ┌──────────────┐
  Result  │ Retry with   │
          │ Exp Backoff  │
          └──────┬───────┘
                 │
          ┌──────┴───────┐
       Success        Max Retries
          │           Exhausted
          ↓               ↓
       Return      ┌──────────┐
       Result      │ Fallback │
                   │ Response │
                   └──────────┘
```

## Implementation Details

### Exponential Backoff

```python
# Retry delays: 1s, 2s, 4s, 8s, 16s...
delay = initial_delay * (base ^ attempt)

# With jitter (randomness):
delay = delay * random(0.5, 1.5)
```

**Why exponential?**
- Quick recovery for transient errors
- Avoids overwhelming recovering service
- Gives time for issues to resolve

**Why jitter?**
- Prevents thundering herd
- Spreads retry load over time
- Reduces system-wide spikes

### Circuit Breaker States

```
CLOSED (Normal)
   │
   │ failures >= threshold
   ↓
OPEN (Failing)
   │
   │ timeout period elapsed
   ↓
HALF_OPEN (Testing)
   │
   ├─ success → CLOSED
   └─ failure → OPEN
```

**State transitions:**
- **CLOSED → OPEN**: Too many consecutive failures
- **OPEN → HALF_OPEN**: Timeout period passes
- **HALF_OPEN → CLOSED**: Service recovered
- **HALF_OPEN → OPEN**: Still failing

### Fallback Strategies

**1. Cached Response**
```python
if primary_fails():
    return cache.get(question)
```

**2. Degraded Service**
```python
if primary_fails():
    return simple_answer_without_context()
```

**3. Error Message**
```python
if primary_fails():
    return "Service temporarily unavailable"
```

## Usage

### Basic Usage
```python
from day26_rag_error_handling import resilient_rag

# Make resilient request
response = resilient_rag("What is ReAct?")

if response.status == "success":
    print(f"Answer: {response.answer.answer}")
    print(f"Retries: {response.retries_attempted}")
elif response.status == "failed":
    if response.fallback_used:
        print(f"Fallback: {response.answer.answer}")
    else:
        print(f"Error: {response.error_message}")
```

### Custom Retry Configuration
```python
from day26_rag_error_handling import ResilientExecutor, RetryConfig

executor = ResilientExecutor(
    retry_config=RetryConfig(
        max_retries=5,
        initial_delay_seconds=0.5,
        max_delay_seconds=30.0,
        exponential_base=2.0,
        jitter=True
    )
)
```

### Circuit Breaker Configuration
```python
from day26_rag_error_handling import CircuitBreakerConfig

circuit_config = CircuitBreakerConfig(
    failure_threshold=3,  # Open after 3 failures
    success_threshold=2,  # Close after 2 successes
    timeout_seconds=60    # Test recovery after 60s
)
```

### Check Circuit Status
```python
status = generation_executor.circuit_breaker.get_status()

print(f"State: {status['state']}")
print(f"Failures: {status['failure_count']}")
```

## Sample Output

```
🛡️  RAG System with Error Handling & Retries

==================================================================
TEST 1: Normal Operation
==================================================================

==================================================================
❓ Question: What is ReAct?
==================================================================

✅ Status: SUCCESS

💡 Answer: ReAct combines reasoning and acting in iterative loops.
📊 Confidence: HIGH
📍 Source: rag

🔄 Resilience:
   Retries: 0
   Duration: 1245.3ms
   Circuit: closed
==================================================================

==================================================================
TEST 3: Transient Failure (Retries)
==================================================================

🧪 Simulating transient failure...
⚠️  Attempt 1 failed: Simulated transient error
   Retrying in 1.2s...
⚠️  Attempt 2 failed: Simulated transient error
   Retrying in 2.8s...

==================================================================
❓ Question: How does agent memory work?
==================================================================

✅ Status: SUCCESS

💡 Answer: Agent memory stores intermediate reasoning steps.
📊 Confidence: HIGH
📍 Source: rag

🔄 Resilience:
   Retries: 2
   Duration: 5127.8ms
   Circuit: closed
==================================================================

==================================================================
🔌 CIRCUIT BREAKER STATUS
==================================================================

Retrieval Circuit:
   State: closed
   Failures: 0
   Successes: 3

Generation Circuit:
   State: closed
   Failures: 0
   Successes: 3
==================================================================
```

## Error Handling Patterns

### Pattern 1: Transient Errors (Retry)
```python
# Network blip, temporary overload, etc.
try:
    result = api_call()
except TransientError:
    # Retry with exponential backoff
    for attempt in range(max_retries):
        wait(exponential_delay(attempt))
        result = api_call()
```

### Pattern 2: Persistent Errors (Circuit Breaker)
```python
# Service is down, database unreachable, etc.
if circuit_breaker.is_open():
    # Fail fast, don't retry
    return fallback_response()

try:
    result = api_call()
    circuit_breaker.record_success()
except:
    circuit_breaker.record_failure()
```

### Pattern 3: Validation Errors (No Retry)
```python
# Bad input, invalid format, etc.
if validation_error:
    # Don't retry - won't help
    return error_message("Invalid input")
```

## Production Considerations

### When to Retry

**✅ Retry these:**
- Network timeouts
- Rate limit errors (with backoff)
- 5xx server errors
- Connection refused
- Temporary overload

**❌ Don't retry these:**
- 4xx client errors (bad request)
- Authentication failures
- Validation errors
- Out of quota (permanent)
- Explicit "do not retry" responses

### Retry Budgets

Set maximum total retry time:
```python
total_budget = 30  # seconds
start = time.time()

while time.time() - start < total_budget:
    try:
        return execute()
    except:
        wait_with_backoff()

# Budget exhausted
return fallback()
```

### Monitoring

Track these metrics:
- Retry rate (% of requests that retry)
- Success after retry rate
- Circuit breaker state transitions
- Average attempts per request
- P95 latency including retries

### Alert Thresholds

```python
alerts = {
    "retry_rate > 10%": "Many transient failures",
    "circuit_open > 5min": "Service degraded",
    "fallback_rate > 5%": "Primary failing often",
    "avg_attempts > 1.5": "Unstable system"
}
```

## Advanced Patterns

### Bulkhead Pattern

Isolate failures to prevent cascades:
```python
# Separate circuit breakers for each component
retrieval_breaker = CircuitBreaker("retrieval")
generation_breaker = CircuitBreaker("generation")
reranking_breaker = CircuitBreaker("reranking")

# Failure in one doesn't affect others
```

### Timeout Pattern

Set strict deadlines:
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)  # 5 second timeout

try:
    result = long_running_operation()
finally:
    signal.alarm(0)  # Cancel timeout
```

### Adaptive Retries

Adjust retry behavior based on error patterns:
```python
class AdaptiveRetry:
    def __init__(self):
        self.recent_success_rate = 1.0

    def should_retry(self, attempt):
        # Fewer retries when success rate is low
        max_retries = int(3 * self.recent_success_rate)
        return attempt < max_retries
```

## Testing

### Simulate Failures
```python
def test_retry_logic():
    # Make function fail first 2 times
    call_count = [0]

    def failing_func():
        call_count[0] += 1
        if call_count[0] <= 2:
            raise Exception("Simulated error")
        return "success"

    # Should succeed on 3rd attempt
    result = execute_with_retry(failing_func)
    assert result == "success"
    assert call_count[0] == 3
```

### Test Circuit Breaker
```python
def test_circuit_breaker():
    breaker = CircuitBreaker(failure_threshold=2)

    # Trigger failures
    breaker.record_failure()
    breaker.record_failure()

    # Circuit should be open
    assert breaker.state == CircuitState.OPEN

    # Requests should be rejected
    can_execute, reason = breaker.can_execute()
    assert not can_execute
```

## Anti-Patterns to Avoid

### ❌ Don't: Retry Forever
```python
# Bad: Infinite retries
while True:
    try:
        return api_call()
    except:
        time.sleep(1)
```

### ✅ Do: Set Max Retries
```python
# Good: Limited retries
for attempt in range(MAX_RETRIES):
    try:
        return api_call()
    except:
        if attempt < MAX_RETRIES - 1:
            backoff.wait(attempt)

raise Exception("Max retries exhausted")
```

### ❌ Don't: Fixed Delay Retries
```python
# Bad: Always wait 1 second
for attempt in range(3):
    try:
        return api_call()
    except:
        time.sleep(1)  # Thundering herd!
```

### ✅ Do: Exponential Backoff with Jitter
```python
# Good: Exponential + jitter
for attempt in range(3):
    try:
        return api_call()
    except:
        delay = (2 ** attempt) * random.uniform(0.5, 1.5)
        time.sleep(delay)
```

### ❌ Don't: Retry Without Circuit Breaker
```python
# Bad: Keep retrying even when service is down
def call_api():
    for attempt in range(10):
        try:
            return api()
        except:
            continue  # Keep trying!
```

### ✅ Do: Use Circuit Breaker
```python
# Good: Fail fast when service is down
if circuit_breaker.is_open():
    return fallback_response()

result = call_api_with_retry()
```

## Next Steps

### Day 27: API Design
Build a production API:
- FastAPI endpoints
- Async request handling
- API documentation

### Future Enhancements
1. **Distributed circuit breakers** - Share state across servers
2. **Adaptive timeouts** - Adjust based on P95 latency
3. **Retry budgets** - Limit total retry time
4. **Fallback chains** - Multiple fallback strategies
5. **Chaos engineering** - Deliberately inject failures

## Files
- `day26_rag_error_handling.py` - Main implementation

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

---

**Production rule:**
> "Everything fails. Plan for it."

Error handling isn't optional—it's the difference between a demo and production. 🛡️
