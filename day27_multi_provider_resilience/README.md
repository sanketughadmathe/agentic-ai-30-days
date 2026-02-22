# Day 27: Multi-Provider Resilience

## Overview
Production RAG systems can't rely on a single LLM provider. This implementation adds intelligent failover between multiple providers with health tracking, automatic cooldowns, and tier-based degradation.

**Inspired by real production patterns from:**
- [@thedarshanjoshi](https://twitter.com/thedarshanjoshi): "Rate limits aren't just throttling—they're failover signals"
- [@AustenMakers](https://twitter.com/AustenMakers): "We bench providers for 5 mins after 2 errors in 10s"

## What This Builds
A multi-provider RAG system with:
- **Automatic failover** - Switch providers on failure
- **Health tracking** - Monitor provider reliability
- **Provider cooldowns** - Bench failing providers temporarily
- **Tier degradation** - Premium → Standard → Budget
- **Rate limit handling** - Treat as failover signal, not retry
- **OpenRouter integration** - Access 100+ models with one API key

**Current setup:**
- Primary: Gemini 2.0 Flash (premium tier)
- Secondary: Arcee Trinity via OpenRouter (standard tier, free)
- Demonstrates cross-provider failover with zero config

## Key Concepts

### Why Multi-Provider Matters

**Single provider risks:**
- Provider outage = your system down
- Rate limits = service degradation
- API changes = production breaks
- Cost optimization = impossible

**Multi-provider benefits:**
- High availability (99.9%+)
- Cost optimization (failover to cheaper)
- Performance optimization (fastest provider)
- Vendor independence

### The Three Patterns

**1. Failover**
```
Primary fails → Try secondary → Try tertiary
```

**2. Load Balancing**
```
Route 80% to fast/cheap → 20% to slow/accurate
```

**3. Tier Degradation**
```
Premium (GPT-4) → Standard (GPT-3.5) → Budget (local)
```

This implementation focuses on **Failover + Tier Degradation**.

## Architecture

```
┌─────────────────────────────────────────┐
│         User Query                      │
└──────────────┬──────────────────────────┘
               │
               ↓
      ┌────────────────┐
      │ Provider       │
      │ Manager        │
      └────────┬───────┘
               │
      ┌────────┴────────────────────┐
      │ Get Available Providers     │
      │ (sorted by tier + health)   │
      └────────┬────────────────────┘
               │
      ┌────────┴─────────┐
      │                  │
   Premium           Standard
   (GPT-4)          (GPT-3.5)
      │                  │
   ┌──┴──┐            ┌──┴──┐
Success Fail      Success Fail
   │      │            │      │
Return  Try         Return  Try
Result  Next        Result  Budget
        Provider            Model
```

### Provider States

```
HEALTHY
   │
   │ failures++
   ↓
DEGRADED (warning)
   │
   │ consecutive_failures >= max_retries
   ↓
COOLDOWN (5 mins)
   │
   │ timeout elapsed
   ↓
DEGRADED (testing)
   │
   │ success
   ↓
HEALTHY
```

## Implementation Details

### Provider Health Tracking

```python
class ProviderHealth:
    status: ProviderStatus  # healthy/degraded/cooldown
    consecutive_failures: int
    success_rate: float  # successful / total
    avg_latency_ms: float
    cooldown_until: datetime | None
```

Tracks:
- **Success rate** - For intelligent routing
- **Latency** - For performance optimization
- **Consecutive failures** - Triggers cooldown
- **Cooldown timer** - When provider recovers

### Cooldown Pattern

```python
# After N consecutive failures
if consecutive_failures >= max_retries:
    cooldown_until = now + 5 minutes
    status = COOLDOWN

# Don't try provider during cooldown
if status == COOLDOWN and now < cooldown_until:
    skip_provider()

# After cooldown expires
if now >= cooldown_until:
    status = DEGRADED  # Test carefully
    cooldown_until = None
```

**Why cooldown?**
- Prevents retry storms
- Gives provider time to recover
- Saves API costs on known-bad providers
- Reduces cascading failures

### Rate Limits as Failover Signals

```python
try:
    response = call_openai()
except RateLimitError:
    # Don't retry same provider!
    # Treat as signal to failover
    failover_to_next_provider()
```

**Key insight:** Rate limits mean "use a different provider", not "try again later".

### Tier-Based Selection

```python
TIERS = [
    ModelTier.PREMIUM,   # GPT-4, Claude Opus ($$$)
    ModelTier.STANDARD,  # GPT-3.5, Claude Sonnet ($$)
    ModelTier.BUDGET,    # Local models, smaller LLMs ($)
    ModelTier.FALLBACK   # Cached/static responses (free)
]

# Try premium first, degrade on failure
for tier in TIERS:
    providers = get_available_providers(tier)
    for provider in providers:
        try:
            return call_provider(provider)
        except:
            continue  # Try next
```

## Usage

### Basic Usage
```python
from day27_multi_provider_resilience import multi_provider_rag

# Automatic failover
response = multi_provider_rag("What is ReAct?")

if response.status == "success":
    print(f"Answer: {response.answer.answer}")
    print(f"Provider: {response.provider_used}")
    print(f"Failovers: {response.failovers_attempted}")
```

### Add Custom Providers
```python
from day27_multi_provider_resilience import ProviderConfig, ModelTier

custom_providers = [
    # OpenAI via direct API
    ProviderConfig(
        name="openai-gpt4",
        model="gpt-4-turbo",
        tier=ModelTier.PREMIUM,
        api_key_env="OPENAI_API_KEY",
        cost_per_1k_tokens=0.01,
        max_retries=2,
        cooldown_minutes=5
    ),

    # Anthropic via OpenRouter
    ProviderConfig(
        name="claude-sonnet",
        model="anthropic/claude-3.5-sonnet",
        tier=ModelTier.STANDARD,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        cost_per_1k_tokens=0.003,
        max_retries=3,
        cooldown_minutes=3
    ),

    # Free models via OpenRouter
    ProviderConfig(
        name="llama-free",
        model="meta-llama/llama-3.2-3b-instruct:free",
        tier=ModelTier.BUDGET,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        cost_per_1k_tokens=0.0,
        max_retries=5,
        cooldown_minutes=2
    )
]
```

**Note:** OpenRouter provides a unified API for 100+ models, including free tiers. This makes it easy to add multiple providers without managing separate API keys.

### Check Provider Health
```python
health = provider_manager.get_health_report()

for name, metrics in health.items():
    print(f"{name}: {metrics['status']} ({metrics['success_rate']})")
```

### Force Specific Tier
```python
# Use only budget tier (for cost savings)
response = multi_provider_rag(
    "What is ReAct?",
    preferred_tier=ModelTier.BUDGET
)
```

## Sample Output

```
🌐 Multi-Provider RAG with Intelligent Failover

==================================================================
TEST 1: Normal Operation
==================================================================

🔄 Trying provider: gemini-flash (tier: premium)

==================================================================
❓ Question: What is ReAct?
==================================================================

✅ Status: SUCCESS

💡 Answer: ReAct combines reasoning and acting in iterative loops.
📊 Confidence: HIGH
🤖 Source: gemini-flash (premium)

🔄 Provider Journey:
   Used: gemini-flash
   Tier: premium
   Tried: gemini-flash
   Failovers: 0
   Latency: 1245.3ms
==================================================================

==================================================================
TEST 2: Provider Failure → Automatic Failover
==================================================================

🧪 Simulating 2 failures on 'gemini-flash'
   Failure 1 recorded
   Failure 2 recorded

⏸️  Provider 'gemini-flash' entering 5min cooldown after 2 failures

   Status: cooldown
   Cooldown until: 2026-02-23T11:22:00

🔄 Trying provider: arcee-trinity (tier: standard)

==================================================================
❓ Question: What improves retrieval precision?
==================================================================

✅ Status: SUCCESS

💡 Answer: Reranking improves retrieval precision.
📊 Confidence: HIGH
🤖 Source: arcee-trinity (standard)

🔄 Provider Journey:
   Used: arcee-trinity
   Tier: standard
   Tried: arcee-trinity
   Failovers: 0
   Latency: 987.5ms
==================================================================

==================================================================
🏥 PROVIDER HEALTH DASHBOARD
==================================================================

✅ GEMINI-FLASH (premium)
   Status: healthy
   Success Rate: 100.0%
   Avg Latency: 1245.3ms
   Total Requests: 1

⏸️  GEMINI-FLASH (premium)
   Status: cooldown
   Success Rate: 0.0%
   Avg Latency: 0.0ms
   Total Requests: 0
   Consecutive Failures: 0
   Cooldown Until: 2026-02-23T11:22:00

✅ ARCEE-TRINITY (standard)
   Status: healthy
   Success Rate: 100.0%
   Avg Latency: 987.5ms
   Total Requests: 2
==================================================================

💡 Key Insights:
==================================================================
• Rate limits trigger automatic failover (not just retries)
• Failed providers enter cooldown to prevent retry storms
• System degrades gracefully through tiers
• Health tracking enables smart routing decisions
==================================================================
```

## Production Patterns

### Pattern 1: Cost Optimization
```python
# Route cheap queries to budget models
if is_simple_query(question):
    response = multi_provider_rag(question, preferred_tier=ModelTier.BUDGET)
else:
    response = multi_provider_rag(question)  # Use premium
```

### Pattern 2: Geographic Routing
```python
providers_by_region = {
    "us-east": ["openai-us", "anthropic-us"],
    "eu-west": ["openai-eu", "mistral-eu"],
    "asia": ["gemini-asia", "claude-asia"]
}

# Route to closest region
region = get_user_region(user_id)
available = providers_by_region[region]
```

### Pattern 3: A/B Testing
```python
# Route 10% to new provider
import random

if random.random() < 0.1:
    response = call_provider("new-experimental-model")
else:
    response = multi_provider_rag(question)

# Track performance difference
```

### Pattern 4: SLA-Based Routing
```python
# Premium users get premium models
if user.tier == "enterprise":
    response = multi_provider_rag(question, preferred_tier=ModelTier.PREMIUM)
else:
    response = multi_provider_rag(question, preferred_tier=ModelTier.STANDARD)
```

## Monitoring & Alerts

### Metrics to Track
```python
metrics = {
    "provider_availability": "% time each provider is available",
    "failover_rate": "% requests that require failover",
    "avg_failovers_per_request": "average provider switches",
    "tier_distribution": "% requests by tier (premium/standard/budget)",
    "cost_per_request": "average cost across all providers"
}
```

### Alert Thresholds
```python
alerts = {
    "provider_cooldown > 5min": "Provider degraded",
    "failover_rate > 20%": "Primary provider unreliable",
    "success_rate < 95%": "System-wide issues",
    "avg_failovers > 2": "Multiple providers failing"
}
```

## Advanced Patterns

### Circuit Breaker + Cooldown
```python
class ProviderWithCircuit:
    def __init__(self):
        self.circuit = CircuitBreaker()
        self.cooldown_manager = CooldownManager()

    def call(self):
        # Check circuit first
        if self.circuit.is_open():
            raise Exception("Circuit open")

        # Check cooldown second
        if self.cooldown_manager.is_in_cooldown():
            raise Exception("In cooldown")

        # Execute
        try:
            result = execute()
            self.circuit.record_success()
            return result
        except:
            self.circuit.record_failure()
            self.cooldown_manager.start_cooldown()
            raise
```

### Adaptive Routing
```python
class AdaptiveRouter:
    def select_provider(self, question):
        # Route based on recent performance
        providers = sorted(
            self.providers,
            key=lambda p: (
                p.recent_success_rate,  # Higher better
                -p.recent_latency       # Lower better
            ),
            reverse=True
        )
        return providers[0]
```

### Cost-Aware Failover
```python
def cost_aware_failover(question, budget_per_request):
    total_cost = 0

    for provider in sorted_by_cost():
        if total_cost + provider.cost > budget_per_request:
            continue  # Skip, too expensive

        try:
            response = call_provider(provider)
            return response
        except:
            total_cost += provider.cost  # Still charged
            continue

    raise Exception("Budget exhausted")
```

## Testing

### Simulate Provider Outages
```python
def test_failover():
    # Simulate primary down
    simulate_provider_failure("primary", num_failures=3)

    # Should automatically use secondary
    response = multi_provider_rag("test question")

    assert response.provider_used == "secondary"
    assert response.failovers_attempted == 1
```

### Load Testing
```python
import concurrent.futures

def load_test(num_requests=100):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(multi_provider_rag, f"Question {i}")
            for i in range(num_requests)
        ]

        results = [f.result() for f in futures]

    # Analyze distribution
    provider_usage = {}
    for r in results:
        provider_usage[r.provider_used] = provider_usage.get(r.provider_used, 0) + 1

    return provider_usage
```

## Cost Analysis

### Example Cost Breakdown
```python
# 1000 requests/day
# 50% use premium (GPT-4: $0.01/1k tokens)
# 30% use standard (GPT-3.5: $0.002/1k tokens)
# 20% use budget (local: $0)

daily_cost = (
    500 * 0.01 +  # Premium
    300 * 0.002 + # Standard
    200 * 0       # Budget
) = $5.60/day = $168/month
```

### Cost Optimization Strategies
1. Route simple queries to budget models
2. Cache aggressively (Day 23)
3. Use streaming for long responses
4. Batch requests when possible
5. Monitor and optimize context sizes

## Next Steps

### Day 28: Deployment
Package everything for production:
- Docker containers
- Environment configuration
- Deployment scripts

### Future Enhancements
1. **Smart routing** - ML-based provider selection
2. **Global load balancing** - Route by geography
3. **Cost budgets** - Hard limits per user/day
4. **Quality scoring** - Track answer quality by provider
5. **Fallback chains** - More than 3 tiers

## Files
- `day27_multi_provider_resilience.py` - Main implementation

## Dependencies
```bash
pip install langchain langchain-openai langchain-huggingface \
    langchain-community faiss-cpu python-dotenv pydantic
```

## Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
TOKENIZERS_PARALLELISM=false
```

### Get OpenRouter API Key
1. Sign up at [openrouter.ai](https://openrouter.ai/)
2. Get your API key from settings
3. Add to `.env` file

OpenRouter provides access to multiple models (including free tiers) through a unified API.

## Learn More
- [OpenRouter](https://openrouter.ai/) - Unified API for 100+ LLMs
- [AWS Multi-Region Patterns](https://aws.amazon.com/blogs/architecture/disaster-recovery-dr-architecture-on-aws-part-iv-multi-site-active-active/)
- [Cloudflare Load Balancing](https://developers.cloudflare.com/load-balancing/)
- [Stripe API Reliability](https://stripe.com/blog/api-versioning)
- [Netflix Hystrix](https://github.com/Netflix/Hystrix) (circuit breaker library)

### Why OpenRouter?

**Single API key, multiple providers:**
- Access GPT-4, Claude, Llama, Mistral, etc.
- Free tier models available
- Automatic failover built-in
- Usage tracking and analytics
- No need to manage multiple API keys

**Perfect for multi-provider setups** - add/remove models without code changes.

## Credits

This implementation was directly inspired by production insights shared on Twitter:

**[@thedarshanjoshi](https://twitter.com/thedarshanjoshi):**
> "Rate limits aren't just throttling—they're failover signals. We use rate limit hits to trigger model switching (GPT-4 → Claude → local)."

**[@AustenMakers](https://twitter.com/AustenMakers):**
> "We added a 'cooldown' state in Redis. If a provider errors twice in 10s, we bench them for 5 mins automatically."

Building in public works—you get mentorship from people solving real production problems. 🙏

---

**Production rule:**
> "A single point of failure is a single point of failure."

Multi-provider resilience isn't optional—it's table stakes for production. 🌐
