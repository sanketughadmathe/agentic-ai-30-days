"""
Production RAG API - FastAPI service with multi-provider resilience
"""

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -----------------------------
# Models
# -----------------------------
class QueryRequest(BaseModel):
    """Request model for /query endpoint."""

    question: str = Field(
        ..., min_length=1, max_length=500, description="User's question"
    )
    user_id: str | None = Field(default=None, description="Optional user identifier")
    preferred_tier: Literal["premium", "standard", "budget"] | None = Field(
        default=None, description="Preferred model tier"
    )


class Answer(BaseModel):
    """Generated answer."""

    answer: str
    confidence: str
    source: str


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""

    status: str
    answer: Answer | None
    provider_used: str | None
    tier_used: str | None
    latency_ms: float
    timestamp: str
    error: str | None = None


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str
    version: str
    uptime_seconds: float
    providers: dict[str, dict]


class MetricsResponse(BaseModel):
    """Response model for /metrics endpoint."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    providers: dict[str, dict]


# -----------------------------
# Provider Configuration
# -----------------------------
class ProviderConfig(BaseModel):
    name: str
    model: str
    tier: str
    api_key_env: str
    base_url: str | None = None
    cost_per_1k_tokens: float
    max_retries: int = 2
    cooldown_minutes: int = 5


PROVIDERS = [
    ProviderConfig(
        name="gemini-flash",
        model="gemini-3-flash-preview",
        tier="premium",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        cost_per_1k_tokens=0.002,
    ),
    ProviderConfig(
        name="arcee-trinity",
        model="arcee-ai/trinity-large-preview:free",
        tier="standard",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        cost_per_1k_tokens=0.0,
    ),
]


# -----------------------------
# Simple Provider Manager
# -----------------------------
class SimpleProviderManager:
    """Lightweight provider manager for API."""

    def __init__(self, providers: list[ProviderConfig]):
        self.providers = {p.name: p for p in providers}
        self.health = {
            name: {
                "status": "healthy",
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
            }
            for name in self.providers
        }

    def get_available_providers(self, tier: str | None = None) -> list[str]:
        """Get available providers sorted by tier."""
        tier_order = {"premium": 0, "standard": 1, "budget": 2}
        available = [
            name
            for name, config in self.providers.items()
            if tier is None or config.tier == tier
        ]
        return sorted(available, key=lambda n: tier_order[self.providers[n].tier])

    def record_success(self, provider_name: str):
        """Record successful request."""
        self.health[provider_name]["total_requests"] += 1
        self.health[provider_name]["successful_requests"] += 1

    def record_failure(self, provider_name: str):
        """Record failed request."""
        self.health[provider_name]["total_requests"] += 1
        self.health[provider_name]["failed_requests"] += 1


# -----------------------------
# Global State
# -----------------------------
class AppState:
    """Application state."""

    def __init__(self):
        self.vectorstore = None
        self.provider_manager = None
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies = []


state = AppState()


# -----------------------------
# Lifespan Management
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""

    print("🚀 Starting RAG API...")

    # Initialize embeddings and vector store
    print("📦 Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("📚 Building vector store...")
    documents = [
        Document(
            page_content="ReAct combines reasoning and acting in iterative loops."
        ),
        Document(page_content="Agent memory stores intermediate reasoning steps."),
        Document(page_content="Structured outputs enforce deterministic contracts."),
        Document(page_content="Reranking improves retrieval precision."),
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
    chunks = splitter.split_documents(documents)

    state.vectorstore = FAISS.from_documents(chunks, embeddings)
    state.provider_manager = SimpleProviderManager(PROVIDERS)

    print("✅ RAG API ready!")

    yield

    print("👋 Shutting down RAG API...")


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Production RAG API",
    description="Multi-provider RAG system with resilience and observability",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Middleware
# -----------------------------
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    """Track request metrics."""
    start_time = time.time()

    response = await call_next(request)

    latency = (time.time() - start_time) * 1000
    state.latencies.append(latency)

    # Keep only last 1000 latencies
    if len(state.latencies) > 1000:
        state.latencies = state.latencies[-1000:]

    return response


# -----------------------------
# Helper Functions
# -----------------------------
def get_llm_client(provider_name: str) -> ChatOpenAI:
    """Create LLM client for a provider."""
    config = state.provider_manager.providers[provider_name]

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise ValueError(f"API key not found: {config.api_key_env}")

    kwargs = {"model": config.model, "api_key": api_key, "temperature": 0}

    if config.base_url:
        kwargs["base_url"] = config.base_url

    return ChatOpenAI(**kwargs)


def execute_rag(question: str, preferred_tier: str | None = None) -> QueryResponse:
    """Execute RAG pipeline with multi-provider failover."""
    start_time = time.time()

    try:
        # Retrieve context
        docs = state.vectorstore.similarity_search(question, k=3)
        context = "\n".join([f"- {d.page_content}" for d in docs])

        # Get available providers
        available = state.provider_manager.get_available_providers(tier=preferred_tier)

        if not available:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No providers available",
            )

        # Try each provider
        last_error = None
        for provider_name in available:
            try:
                config = state.provider_manager.providers[provider_name]
                llm = get_llm_client(provider_name)
                llm_structured = llm.with_structured_output(Answer)

                prompt = f"""Answer using only this context:

                    {context}

                    Question: {question}

                    Provide your answer with confidence level.
                """

                answer = llm_structured.invoke(prompt)
                answer.source = f"{provider_name} ({config.tier})"

                # Record success
                state.provider_manager.record_success(provider_name)
                state.successful_requests += 1

                latency = (time.time() - start_time) * 1000

                return QueryResponse(
                    status="success",
                    answer=answer,
                    provider_used=provider_name,
                    tier_used=config.tier,
                    latency_ms=latency,
                    timestamp=datetime.now().isoformat(),
                )

            except Exception as e:
                last_error = str(e)
                state.provider_manager.record_failure(provider_name)
                continue

        # All providers failed
        state.failed_requests += 1
        latency = (time.time() - start_time) * 1000

        return QueryResponse(
            status="failed",
            answer=None,
            provider_used=None,
            tier_used=None,
            latency_ms=latency,
            timestamp=datetime.now().isoformat(),
            error=f"All providers failed: {last_error}",
        )

    except Exception as e:
        state.failed_requests += 1
        latency = (time.time() - start_time) * 1000

        return QueryResponse(
            status="error",
            answer=None,
            provider_used=None,
            tier_used=None,
            latency_ms=latency,
            timestamp=datetime.now().isoformat(),
            error=str(e),
        )


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Production RAG API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check endpoint."""
    uptime = time.time() - state.start_time

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime,
        providers={
            name: health for name, health in state.provider_manager.health.items()
        },
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["General"])
async def metrics():
    """Metrics endpoint."""
    avg_latency = (
        sum(state.latencies) / len(state.latencies) if state.latencies else 0.0
    )

    return MetricsResponse(
        total_requests=state.total_requests,
        successful_requests=state.successful_requests,
        failed_requests=state.failed_requests,
        avg_latency_ms=avg_latency,
        providers=state.provider_manager.health,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Query the RAG system.

    - **question**: The question to answer
    - **user_id**: Optional user identifier for tracking
    - **preferred_tier**: Optional tier preference (premium/standard/budget)
    """
    state.total_requests += 1

    try:
        response = execute_rag(request.question, request.preferred_tier)
        return response
    except HTTPException:
        raise
    except Exception as e:
        state.failed_requests += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@app.get("/providers", tags=["General"])
async def list_providers():
    """List available providers and their status."""
    return {
        name: {
            "model": config.model,
            "tier": config.tier,
            "cost_per_1k_tokens": config.cost_per_1k_tokens,
            "health": state.provider_manager.health[name],
        }
        for name, config in state.provider_manager.providers.items()
    }


# -----------------------------
# Error Handlers
# -----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
        },
    )


# -----------------------------
# Run with: uvicorn rag_api:app --reload
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
