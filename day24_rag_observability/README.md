# Day 24: RAG Observability

## Overview
Production RAG systems need instrumentation to track performance, identify bottlenecks, and maintain reliability. This implementation adds comprehensive observability to a RAG pipeline with metrics tracking, health monitoring, and latency analysis.

## What This Builds
A RAG system with full observability that tracks:
- **Latency metrics** - Retrieval, generation, and end-to-end timing
- **Success rates** - Track failures and error patterns
- **Cache performance** - Monitor cache hit rates
- **System health** - Aggregate metrics over time windows
- **P95 latencies** - Identify tail latencies that hurt UX

## Key Concepts

### Why Observability Matters
> "You can't fix what you can't see."

Without observability, you're flying blind:
- Is retrieval slow or is generation slow?
- What's the P95 latency users experience?
- Did that optimization actually help?
- Where should you focus improvement efforts?

### Metrics Collected

**Per-Query Metrics:**
- Retrieval latency (ms)
- Generation latency (ms)
- Total end-to-end latency (ms)
- Number of documents retrieved
- Cache hit/miss status
- Answer quality indicators (confidence, grounding)
- Error status

**Aggregate Health Metrics:**
- Total queries processed
- Success rate (% successful)
- Cache hit rate (% cached)
- Average latencies (retrieval, generation, total)
- P95 latencies (95th percentile - what most users experience)
- Failed query count

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 RAG Pipeline                        │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │Retrieval │→ │Generation│→ │  Answer  │         │
│  └──────────┘  └──────────┘  └──────────┘         │
│       ↓              ↓              ↓              │
│  ┌──────────────────────────────────────┐          │
│  │       RAGObserver                    │          │
│  │  • Start/end span tracking           │          │
│  │  • Latency measurement               │          │
│  │  • Metrics persistence (JSONL)       │          │
│  └──────────────────────────────────────┘          │
│                     ↓                               │
│  ┌──────────────────────────────────────┐          │
│  │       rag_metrics.jsonl              │          │
│  │  (timestamped metrics for analysis)  │          │
│  └──────────────────────────────────────┘          │
│                     ↓                               │
│  ┌──────────────────────────────────────┐          │
│  │       System Health Dashboard        │          │
│  │  • Success rate                      │          │
│  │  • Cache hit rate                    │          │
│  │  • Latency percentiles (avg, P95)    │          │
│  └──────────────────────────────────────┘          │
└─────────────────────────────────────────────────────┘
```

## Implementation Details

### Span Tracking
```python
observer.start_span("retrieval")
docs = vectorstore.similarity_search(question, k=3)
retrieval_latency = observer.end_span("retrieval")
```

Uses wall-clock timing to measure each component independently.

### Metrics Persistence
```python
# Append metrics to JSONL (one JSON object per line)
{
  "timestamp": "2026-02-18T10:30:00",
  "question": "What is ReAct?",
  "retrieval_latency_ms": 145.2,
  "generation_latency_ms": 892.5,
  "total_latency_ms": 1037.7,
  "cache_hit": false,
  "confidence": "HIGH",
  "grounded": true,
  "error": null
}
```

JSONL format allows easy streaming and analysis without loading entire file.

### Health Computation
```python
# Load metrics from last 24 hours
metrics = observer.load_metrics(time_window_hours=24)

# Compute P95 latency
sorted_latencies = sorted([m.total_latency_ms for m in metrics])
p95_index = int(len(sorted_latencies) * 0.95)
p95_latency = sorted_latencies[p95_index]
```

## Usage

### Basic Usage
```python
from day24_rag_observability import observed_rag, observer

# Run query with full tracking
answer, metrics = observed_rag("What is ReAct?")

print(f"Answer: {answer.answer}")
print(f"Total latency: {metrics.total_latency_ms}ms")
print(f"Confidence: {answer.confidence}")
```

### View System Health
```python
# Get aggregate metrics from last 24 hours
health = observer.compute_health(time_window_hours=24)

print(f"Success rate: {health.success_rate:.1%}")
print(f"Cache hit rate: {health.cache_hit_rate:.1%}")
print(f"P95 latency: {health.p95_total_latency_ms}ms")
```

### Analyze Metrics
```python
# Load raw metrics for custom analysis
metrics = observer.load_metrics(time_window_hours=24)

# Find slowest queries
slow_queries = sorted(metrics, key=lambda m: m.total_latency_ms, reverse=True)[:5]

for m in slow_queries:
    print(f"{m.question}: {m.total_latency_ms}ms")
```

## Sample Output

```
🔍 RAG System with Observability

==================================================================
❓ Question: What is ReAct?
==================================================================
💡 Answer: ReAct combines reasoning and acting in iterative loops.
📊 Confidence: HIGH
✓ Grounded: Yes

⏱️  Latency Breakdown:
   Retrieval: 145.2ms
   Generation: 892.5ms
   Total: 1037.7ms

📈 Metadata:
   Candidates: 3
   Answer Length: 56
   Cache Hit: False
   Model: gemini-2.0-flash-exp
==================================================================

📊 SYSTEM HEALTH (Last 24.0h)
==================================================================

📝 Volume:
   Total Queries: 4
   ✅ Successful: 4
   ❌ Failed: 0
   Success Rate: 100.0%

💾 Caching:
   Cache Hit Rate: 25.0%

⏱️  Latency (ms):
   Retrieval:
      Avg: 158.3ms
      P95: 172.1ms
   Generation:
      Avg: 945.2ms
      P95: 1024.8ms
   Total:
      Avg: 1103.5ms
      P95: 1196.9ms
==================================================================
```

## Key Insights

### 1. Latency Attribution
Know exactly where time is spent:
- Retrieval slow? → Check vector DB, try caching embeddings
- Generation slow? → Check model size, consider streaming
- Total slow? → Look at network, consider async processing

### 2. P95 > Average
Average latency is misleading. P95 shows what most users experience:
```
Average: 500ms  (looks good!)
P95: 2000ms     (users are unhappy)
```

Always optimize for P95, not average.

### 3. Cache Impact
Track cache hit rate to measure optimization ROI:
```
Before caching:
- Avg latency: 1200ms
- Cache hit rate: 0%

After caching:
- Avg latency: 450ms (62% reduction)
- Cache hit rate: 65%
```

### 4. Error Patterns
Failed queries reveal edge cases:
- Empty retrievals → Improve chunking/indexing
- Generation timeouts → Add streaming or shorter prompts
- Validation failures → Tighten guardrails

## Production Considerations

### Storage
JSONL files grow over time. Implement rotation:
```python
# Rotate metrics file when it exceeds 100MB
if metrics_file.stat().st_size > 100 * 1024 * 1024:
    metrics_file.rename(f"rag_metrics_{datetime.now().isoformat()}.jsonl")
```

### Monitoring
Send metrics to external systems:
- **Prometheus** - For alerting on latency spikes
- **Grafana** - For dashboards and visualization
- **DataDog/New Relic** - For APM integration

### Alerts
Set up alerts for degradation:
- P95 latency > 2000ms
- Success rate < 95%
- Cache hit rate < 50% (if caching is critical)

## Next Steps

### Day 25: Rate Limiting
Add protection for high-load scenarios:
- Per-user rate limits
- Graceful degradation
- Queue management

### Future Enhancements
1. **Distributed tracing** - Track requests across services
2. **Cost tracking** - Monitor LLM API costs per query
3. **User segmentation** - Track metrics per user/tenant
4. **A/B testing** - Compare different RAG configurations
5. **Anomaly detection** - Auto-detect unusual patterns

## Files
- `day24_rag_observability.py` - Main implementation
- `rag_metrics.jsonl` - Metrics storage (created on first run)

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
- [Observability in distributed systems](https://landing.google.com/sre/sre-book/chapters/monitoring-distributed-systems/)
- [RED method (Rate, Errors, Duration)](https://www.weave.works/blog/the-red-method-key-metrics-for-microservices-architecture/)
- [P95 vs P99 latencies](https://blog.cloudflare.com/a-question-of-timing/)

---

**The oldest rule in engineering:**
> "You can't fix what you can't see."

Observability is the foundation for reliability. 🔍
