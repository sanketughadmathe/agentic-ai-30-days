# Day 28: Production RAG Deployment

## Overview
Package the entire RAG system as a production-ready API with Docker containerization, health checks, metrics, and deployment orchestration.

## What This Builds
A complete production deployment including:
- **FastAPI REST API** - Async endpoints with automatic docs
- **Docker containerization** - Reproducible deployment
- **docker-compose orchestration** - Easy multi-container management
- **Health checks** - Automatic monitoring
- **Metrics endpoint** - Observability
- **Multi-provider support** - Gemini + OpenRouter failover
- **CORS middleware** - Cross-origin support
- **Error handling** - Graceful degradation

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Docker Container                     │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │         FastAPI Application (port 8000)            │  │
│  │                                                    │  │
│  │          ┌────────────┐  ┌──────────────┐          │  │
│  │          │  /query    │  │   /health    │          │  │
│  │          │  /metrics  │  │  /providers  │          │  │
│  │          └────────────┘  └──────────────┘          │  │
│  │                │               │                   │  │
│  │                ↓               ↓                   │  │
│  │       ┌────────────────────────────────────┐       │  │
│  │       │    Multi-Provider RAG System       │       │  │
│  │       │  • Vector store (FAISS)            │       │  │
│  │       │  • Provider manager                │       │  │
│  │       │  • Failover logic                  │       │  │
│  │       └────────────────────────────────────┘       │  │
│  │             │                                      │  │
│  │             ↓                                      │  │
│  │        ┌──────────────┐    ┌──────────────┐        │  │
│  │        │   Gemini     │ ←→ │  OpenRouter  │        │  │
│  │        │   (Primary)  │    │  (Fallback)  │        │  │
│  │        └──────────────┘    └──────────────┘        │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Health Check: curl localhost:8000/health                │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Docker installed
- Docker Compose installed
- API keys (Gemini + OpenRouter)

### Setup

```bash
# 1. Clone/navigate to directory
cd day28_rag_deployment

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env with your API keys
nano .env  # or use your editor

# 4. Build and start
docker-compose up -d

# 5. Check health
curl http://localhost:8000/health

# 6. View docs
open http://localhost:8000/docs
```

**That's it!** Your production RAG API is running.

## API Endpoints

### POST /query
Query the RAG system.

**Request:**
```json
{
  "question": "What is ReAct?",
  "user_id": "optional_user_id",
  "preferred_tier": "premium"
}
```

**Response:**
```json
{
  "status": "success",
  "answer": {
    "answer": "ReAct combines reasoning and acting in iterative loops.",
    "confidence": "HIGH",
    "source": "gemini-flash (premium)"
  },
  "provider_used": "gemini-flash",
  "tier_used": "premium",
  "latency_ms": 1245.3,
  "timestamp": "2026-02-23T10:30:00.000Z"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is ReAct?",
    "preferred_tier": "premium"
  }'
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "providers": {
    "gemini-flash": {
      "status": "healthy",
      "total_requests": 100,
      "successful_requests": 95,
      "failed_requests": 5
    },
    "arcee-trinity": {
      "status": "healthy",
      "total_requests": 20,
      "successful_requests": 20,
      "failed_requests": 0
    }
  }
}
```

### GET /metrics
System metrics.

**Response:**
```json
{
  "total_requests": 120,
  "successful_requests": 115,
  "failed_requests": 5,
  "avg_latency_ms": 987.5,
  "providers": {
    "gemini-flash": {...},
    "arcee-trinity": {...}
  }
}
```

### GET /providers
List available providers.

**Response:**
```json
{
  "gemini-flash": {
    "model": "gemini-2.0-flash-exp",
    "tier": "premium",
    "cost_per_1k_tokens": 0.002,
    "health": {...}
  },
  "arcee-trinity": {
    "model": "arcee-ai/trinity-large-preview:free",
    "tier": "standard",
    "cost_per_1k_tokens": 0.0,
    "health": {...}
  }
}
```

## Docker Commands

### Using Makefile (Recommended)
```bash
# Build image
make build

# Start services
make up

# View logs
make logs

# Stop services
make down

# Restart
make restart

# Test API
make test

# Clean up everything
make clean
```

### Using docker-compose directly
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Restart
docker-compose restart

# View status
docker-compose ps
```

### Using Docker directly
```bash
# Build image
docker build -t rag-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e OPENROUTER_API_KEY=your_key \
  --name rag-api \
  rag-api

# View logs
docker logs -f rag-api

# Stop container
docker stop rag-api

# Remove container
docker rm rag-api
```

## File Structure

```
day28_rag_deployment/
├── day28_rag_api.py      # FastAPI application
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Multi-container orchestration
├── requirements.txt      # Python dependencies
├── .dockerignore        # Files to exclude from image
├── .env.example         # Environment template
├── .env                 # Your API keys (git-ignored)
├── Makefile            # Development commands
└── README.md           # This file
```

## Configuration

### Environment Variables

**Required:**
```bash
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

**Optional:**
```bash
TOKENIZERS_PARALLELISM=false
LOG_LEVEL=info
MAX_WORKERS=4
```

### Port Configuration

Default: `8000`

To change:
```yaml
# docker-compose.yml
ports:
  - "3000:8000"  # Map host:container
```

### Volume Mounts

For development (hot reload):
```yaml
volumes:
  - ./:/app
```

For production (no mount):
```yaml
# Remove volumes section
```

## Production Deployment

### Option 1: Single Server

```bash
# SSH to server
ssh user@your-server.com

# Clone repo
git clone your-repo
cd day28_rag_deployment

# Setup environment
cp .env.example .env
nano .env  # Add API keys

# Start
docker-compose up -d

# Configure reverse proxy (nginx)
# See nginx example below
```

### Option 2: Cloud Platforms

**AWS (ECS):**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login ...
docker build -t rag-api .
docker tag rag-api:latest your-account.dkr.ecr.us-east-1.amazonaws.com/rag-api:latest
docker push ...

# Deploy to ECS
aws ecs create-service ...
```

**Google Cloud (Cloud Run):**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/your-project/rag-api
gcloud run deploy rag-api \
  --image gcr.io/your-project/rag-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**DigitalOcean (App Platform):**
- Connect GitHub repo
- Select Dockerfile
- Add environment variables
- Deploy

### Option 3: Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: gemini
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openrouter
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/rag-api
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/rag-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Monitoring & Observability

### Health Checks

**Docker health check (built-in):**
```bash
docker ps  # Check STATUS column for "healthy"
```

**External monitoring:**
```bash
# Add to cron or monitoring service
*/5 * * * * curl -f http://localhost:8000/health || alert
```

### Metrics Collection

**Prometheus:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rag-api'
    static_configs:
      - targets: ['rag-api:8000']
    metrics_path: '/metrics'
```

### Logging

**View logs:**
```bash
docker-compose logs -f rag-api
```

**Log to file:**
```yaml
# docker-compose.yml
services:
  rag-api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**External logging (e.g., ELK):**
```yaml
logging:
  driver: "syslog"
  options:
    syslog-address: "tcp://logstash:5000"
```

## Testing

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ReAct?"}'

# Metrics
curl http://localhost:8000/metrics

# Providers
curl http://localhost:8000/providers
```

### Load Testing

**Using Apache Bench:**
```bash
ab -n 1000 -c 10 -p query.json -T application/json \
  http://localhost:8000/query
```

**Using wrk:**
```bash
wrk -t4 -c100 -d30s \
  -s query.lua \
  http://localhost:8000/query
```

### Integration Tests

```python
# test_api.py
import requests

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query():
    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": "What is ReAct?"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["answer"] is not None

def test_metrics():
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    assert "total_requests" in response.json()
```

Run with pytest:
```bash
pytest test_api.py -v
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs rag-api

# Common issues:
# - Missing .env file
# - Invalid API keys
# - Port 8000 already in use
```

### API returns 503
```bash
# Check provider health
curl http://localhost:8000/providers

# Verify API keys
docker-compose exec rag-api env | grep API_KEY
```

### Slow responses
```bash
# Check metrics
curl http://localhost:8000/metrics

# Check container resources
docker stats rag-api

# Increase resources in docker-compose.yml:
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

### Permission errors
```bash
# Fix file permissions
chmod +x Makefile
chmod 600 .env
```

## Security Best Practices

### 1. Environment Variables
```bash
# Never commit .env to git
echo ".env" >> .gitignore

# Use secrets management in production
docker secret create gemini_key /path/to/key
```

### 2. API Authentication

Add authentication middleware:
```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/query")
async def query(request: QueryRequest, api_key: str = Depends(get_api_key)):
    ...
```

### 3. Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, query: QueryRequest):
    ...
```

### 4. HTTPS

Use a reverse proxy (nginx) with SSL/TLS:
```bash
certbot --nginx -d api.yourdomain.com
```

## Cost Optimization

### Estimated Costs

**Gemini (primary):**
- $0.002 per 1k tokens
- Average 500 tokens per request
- = $0.001 per request

**If 1000 requests/day:**
- 70% use Gemini (700 requests)
- 30% failover to free tier (300 requests)
- Daily cost: $0.70
- Monthly cost: ~$21

**Cost reduction strategies:**
1. Cache aggressively (50% hit rate = 50% cost reduction)
2. Use free tier when possible
3. Batch similar queries
4. Optimize prompt lengths

## Next Steps

### Day 29: Advanced Patterns
- Self-RAG
- Agentic RAG
- Multi-hop reasoning

### Day 30: Final Integration
- Portfolio project showcase
- Blog post: "30 Days of Production RAG"

### Future Enhancements
1. **Horizontal scaling** - Multiple API instances
2. **Database integration** - PostgreSQL for persistence
3. **User authentication** - JWT tokens
4. **Advanced caching** - Redis integration
5. **Streaming responses** - Server-sent events

## Files

- `day28_rag_api.py` - FastAPI application
- `Dockerfile` - Container definition
- `docker-compose.yml` - Orchestration
- `requirements.txt` - Dependencies
- `.dockerignore` - Build exclusions
- `.env.example` - Environment template
- `Makefile` - Development commands

## Dependencies

All dependencies are in `requirements.txt`:
```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
langchain==0.1.6
langchain-openai==0.0.5
sentence-transformers==2.3.1
faiss-cpu==1.7.4
pydantic==2.6.0
python-dotenv==1.0.1
```

---

**Production rule:**
> "If it's not containerized, it's not production-ready."

This is how you ship RAG systems. 🚀
