"""
Self-RAG: RAG system that evaluates and corrects its own responses

Self-RAG adds a reflection loop where the model:
1. Retrieves context
2. Generates answer
3. Self-evaluates the answer
4. If quality is low, retrieves more context or regenerates
5. Returns best answer after N iterations
"""

import os
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
class RetrievalDecision(str, Enum):
    """Whether to retrieve more context."""

    RETRIEVE = "retrieve"  # Need more context
    NO_RETRIEVE = "no_retrieve"  # Sufficient context


class AnswerQuality(str, Enum):
    """Quality assessment of generated answer."""

    EXCELLENT = "excellent"  # High quality, use as-is
    GOOD = "good"  # Acceptable quality
    POOR = "poor"  # Low quality, regenerate
    UNSUPPORTED = "unsupported"  # Not grounded in context


class RetrievalJudgment(BaseModel):
    """Decision on whether to retrieve more documents."""

    decision: RetrievalDecision
    reasoning: str
    search_queries: list[str] = Field(
        default=[], description="Additional search queries if retrieving"
    )


class Answer(BaseModel):
    """Generated answer."""

    answer: str
    confidence: str
    key_claims: list[str] = Field(description="Main claims in the answer", max_length=5)


class SelfEvaluation(BaseModel):
    """Self-evaluation of generated answer."""

    quality: AnswerQuality
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="How relevant to question"
    )
    completeness_score: float = Field(ge=0.0, le=1.0, description="How complete")
    grounding_score: float = Field(
        ge=0.0, le=1.0, description="How well grounded in context"
    )
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall quality")
    issues: list[str] = Field(default=[], description="Problems identified")
    suggestions: str = Field(description="How to improve if regenerating")


class SelfRAGIteration(BaseModel):
    """Single iteration of Self-RAG."""

    iteration: int
    num_docs_retrieved: int
    answer: Answer
    evaluation: SelfEvaluation
    retrieval_decision: RetrievalJudgment | None = None


class SelfRAGResponse(BaseModel):
    """Complete Self-RAG response with all iterations."""

    question: str
    final_answer: Answer
    final_evaluation: SelfEvaluation
    iterations: list[SelfRAGIteration]
    total_iterations: int
    improvement_achieved: bool


# -----------------------------
# Setup
# -----------------------------
MODEL = "gemini-3-flash-preview"

llm = ChatOpenAI(
    model=MODEL,
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
)

# Vector store
documents = [
    Document(
        page_content="ReAct combines reasoning and acting in iterative loops. It enables agents to dynamically adjust their behavior based on observations."
    ),
    Document(
        page_content="Agent memory stores intermediate reasoning steps, allowing the agent to maintain context across multiple iterations."
    ),
    Document(
        page_content="Structured outputs enforce deterministic contracts between the LLM and the application, ensuring reliable parsing."
    ),
    Document(
        page_content="Reranking improves retrieval precision by re-scoring candidates after initial retrieval, focusing on relevance."
    ),
    Document(
        page_content="Self-RAG adds a reflection mechanism where the model evaluates its own responses and iteratively improves them."
    ),
    Document(
        page_content="Agentic RAG systems can decide when to retrieve more information versus using existing context."
    ),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)


# -----------------------------
# Self-RAG Components
# -----------------------------
def retrieve_context(
    question: str, k: int = 3, exclude_docs: set[str] = None
) -> list[Document]:
    """
    Retrieve context for question.

    Args:
        question: User's question
        k: Number of documents to retrieve
        exclude_docs: Set of document contents to exclude (already retrieved)
    """
    docs = vectorstore.similarity_search(question, k=k * 2)  # Get extra for filtering

    if exclude_docs:
        docs = [d for d in docs if d.page_content not in exclude_docs]

    return docs[:k]


def judge_retrieval_need(
    question: str, current_context: str, previous_eval: SelfEvaluation | None
) -> RetrievalJudgment:
    """
    Decide whether to retrieve more documents.

    Uses LLM to judge if current context is sufficient.
    """
    llm_structured = llm.with_structured_output(RetrievalJudgment)

    eval_info = ""
    if previous_eval:
        eval_info = f"""
            Previous evaluation identified these issues:
            {chr(10).join([f"- {issue}" for issue in previous_eval.issues])}

            Suggestions: {previous_eval.suggestions}
        """

    prompt = f"""You are helping a RAG system decide whether to retrieve more documents.

        Question: {question}

        Current Context:
        {current_context}

        {eval_info}

        Decide:
        - RETRIEVE: If context is insufficient or answer quality was poor
        - NO_RETRIEVE: If context is sufficient to answer well

        If retrieving, suggest specific search queries to get better context.
    """

    return llm_structured.invoke(prompt)


def generate_answer(question: str, context: str) -> Answer:
    """Generate answer from context."""
    llm_structured = llm.with_structured_output(Answer)

    prompt = f"""Answer the question using ONLY the provided context.

        Context:
        {context}

        Question: {question}

        Provide:
        1. Your answer
        2. Confidence level (HIGH/MEDIUM/LOW)
        3. Key claims you're making (for verification)
    """

    return llm_structured.invoke(prompt)


def self_evaluate(question: str, answer: Answer, context: str) -> SelfEvaluation:
    """
    Self-evaluate the generated answer.

    The model critiques its own response.
    """
    llm_structured = llm.with_structured_output(SelfEvaluation)

    prompt = f"""You are evaluating your own answer. Be critical and honest.

        Question: {question}

        Your Answer: {answer.answer}

        Your Key Claims:
        {chr(10).join([f"- {claim}" for claim in answer.key_claims])}

        Context Used:
        {context}

        Evaluate your answer on:
        1. Quality (EXCELLENT/GOOD/POOR/UNSUPPORTED)
        2. Relevance (0.0-1.0): Does it answer the question?
        3. Completeness (0.0-1.0): Is it complete?
        4. Grounding (0.0-1.0): Is it supported by context?
        5. Overall score (0.0-1.0): Combined quality

        Identify specific issues and suggest improvements.
        Be harsh - this is for improving the answer.
    """

    return llm_structured.invoke(prompt)


# -----------------------------
# Self-RAG Main Loop
# -----------------------------
def self_rag(
    question: str,
    max_iterations: int = 3,
    quality_threshold: float = 0.8,
    verbose: bool = True,
) -> SelfRAGResponse:
    """
    Self-RAG with iterative improvement.

    Process:
    1. Retrieve initial context
    2. Generate answer
    3. Self-evaluate
    4. If quality < threshold and iterations remain:
       - Decide if more retrieval needed
       - Retrieve more or regenerate with same context
       - Repeat
    5. Return best answer

    Args:
        question: User's question
        max_iterations: Maximum refinement iterations
        quality_threshold: Minimum acceptable quality score
        verbose: Print iteration details
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"🔄 Self-RAG: {question}")
        print(f"{'=' * 70}\n")

    iterations = []
    retrieved_docs_content = set()
    best_answer = None
    best_evaluation = None
    best_score = 0.0

    # Initial retrieval
    docs = retrieve_context(question, k=3)
    for doc in docs:
        retrieved_docs_content.add(doc.page_content)

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")

        # Current context
        context = "\n\n".join(retrieved_docs_content)

        if verbose:
            print(f"📚 Context: {len(retrieved_docs_content)} documents")

        # Generate answer
        answer = generate_answer(question, context)

        if verbose:
            print(f"💡 Answer: {answer.answer[:100]}...")

        # Self-evaluate
        evaluation = self_evaluate(question, answer, context)

        if verbose:
            print(f"📊 Quality: {evaluation.quality.value}")
            print(f"   Overall Score: {evaluation.overall_score:.2f}")
            print(f"   Relevance: {evaluation.relevance_score:.2f}")
            print(f"   Completeness: {evaluation.completeness_score:.2f}")
            print(f"   Grounding: {evaluation.grounding_score:.2f}")

        # Track iteration
        iter_data = SelfRAGIteration(
            iteration=iteration + 1,
            num_docs_retrieved=len(retrieved_docs_content),
            answer=answer,
            evaluation=evaluation,
        )

        # Update best if this is better
        if evaluation.overall_score > best_score:
            best_answer = answer
            best_evaluation = evaluation
            best_score = evaluation.overall_score

            if verbose:
                print("   ✨ New best score!")

        # Check if quality is acceptable
        if evaluation.overall_score >= quality_threshold:
            if verbose:
                print(
                    f"\n✅ Quality threshold met ({evaluation.overall_score:.2f} >= {quality_threshold})"
                )

            iter_data.retrieval_decision = None
            iterations.append(iter_data)
            break

        # Not last iteration - decide if we need more context
        if iteration < max_iterations - 1:
            retrieval_decision = judge_retrieval_need(question, context, evaluation)
            iter_data.retrieval_decision = retrieval_decision

            if verbose:
                print(f"\n🤔 Retrieval Decision: {retrieval_decision.decision.value}")
                print(f"   Reasoning: {retrieval_decision.reasoning}")

            if retrieval_decision.decision == RetrievalDecision.RETRIEVE:
                # Retrieve more documents
                if retrieval_decision.search_queries:
                    # Use suggested queries
                    for query in retrieval_decision.search_queries[
                        :2
                    ]:  # Limit to 2 queries
                        new_docs = retrieve_context(
                            query, k=2, exclude_docs=retrieved_docs_content
                        )
                        for doc in new_docs:
                            retrieved_docs_content.add(doc.page_content)

                        if verbose:
                            print(
                                f"   📥 Retrieved {len(new_docs)} more docs for: '{query}'"
                            )
                else:
                    # Default retrieval
                    new_docs = retrieve_context(
                        question, k=2, exclude_docs=retrieved_docs_content
                    )
                    for doc in new_docs:
                        retrieved_docs_content.add(doc.page_content)

                    if verbose:
                        print(f"   📥 Retrieved {len(new_docs)} more docs")
            else:
                if verbose:
                    print("   ♻️  Will regenerate with same context")

        iterations.append(iter_data)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"✅ Final Answer (Score: {best_score:.2f})")
        print(f"{'=' * 70}")

    improvement = (
        best_score > iterations[0].evaluation.overall_score
        if len(iterations) > 1
        else False
    )

    return SelfRAGResponse(
        question=question,
        final_answer=best_answer,
        final_evaluation=best_evaluation,
        iterations=iterations,
        total_iterations=len(iterations),
        improvement_achieved=improvement,
    )


# -----------------------------
# Comparison: Regular RAG vs Self-RAG
# -----------------------------
def regular_rag(question: str) -> tuple[Answer, float]:
    """Regular RAG for comparison."""
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    answer = generate_answer(question, context)
    evaluation = self_evaluate(question, answer, context)

    return answer, evaluation.overall_score


def compare_rag_approaches(question: str):
    """Compare Regular RAG vs Self-RAG."""

    print(f"\n{'=' * 70}")
    print("🔬 Comparing RAG Approaches")
    print(f"{'=' * 70}")
    print(f"Question: {question}\n")

    # Regular RAG
    print("1️⃣  Regular RAG (single-shot)")
    print("-" * 70)
    regular_answer, regular_score = regular_rag(question)
    print(f"Answer: {regular_answer.answer}")
    print(f"Score: {regular_score:.2f}")

    # Self-RAG
    print("\n2️⃣  Self-RAG (iterative improvement)")
    print("-" * 70)
    self_rag_response = self_rag(question, max_iterations=3, verbose=False)
    print(f"Answer: {self_rag_response.final_answer.answer}")
    print(f"Score: {self_rag_response.final_evaluation.overall_score:.2f}")
    print(f"Iterations: {self_rag_response.total_iterations}")
    print(f"Improved: {'Yes' if self_rag_response.improvement_achieved else 'No'}")

    # Comparison
    print("\n📊 Comparison")
    print("-" * 70)
    improvement = self_rag_response.final_evaluation.overall_score - regular_score
    print(f"Score Improvement: {improvement:+.2f}")
    print(
        f"Self-RAG is {abs(improvement) / regular_score * 100:.1f}% {'better' if improvement > 0 else 'worse'}"
    )


# -----------------------------
# Pretty Print
# -----------------------------
def print_self_rag_response(response: SelfRAGResponse):
    """Pretty print Self-RAG response."""

    print(f"\n{'=' * 70}")
    print("📝 SELF-RAG COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n❓ Question: {response.question}")
    print("\n💡 Final Answer:")
    print(f"   {response.final_answer.answer}")
    print("\n📊 Final Evaluation:")
    print(f"   Quality: {response.final_evaluation.quality.value}")
    print(f"   Overall Score: {response.final_evaluation.overall_score:.2f}")
    print(f"   Relevance: {response.final_evaluation.relevance_score:.2f}")
    print(f"   Completeness: {response.final_evaluation.completeness_score:.2f}")
    print(f"   Grounding: {response.final_evaluation.grounding_score:.2f}")

    print("\n🔄 Improvement Journey:")
    for iter_data in response.iterations:
        print(
            f"   Iteration {iter_data.iteration}: Score {iter_data.evaluation.overall_score:.2f} ({iter_data.evaluation.quality.value})"
        )

    if response.improvement_achieved:
        initial_score = response.iterations[0].evaluation.overall_score
        final_score = response.final_evaluation.overall_score
        improvement = final_score - initial_score
        print(
            f"\n✨ Improvement: {improvement:+.2f} ({improvement / initial_score * 100:+.1f}%)"
        )
    else:
        print("\n➡️  No improvement over initial answer")

    print(f"{'=' * 70}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Test 1: Self-RAG with verbose output
    print("\n🧪 TEST 1: Self-RAG with Iterative Improvement")
    response1 = self_rag(
        "How does Self-RAG work and why is it better than regular RAG?",
        max_iterations=3,
        quality_threshold=0.85,
        verbose=True,
    )
    print_self_rag_response(response1)

    # Test 2: Comparison
    print("\n\n🧪 TEST 2: Regular RAG vs Self-RAG Comparison")
    compare_rag_approaches("What is ReAct and how does it enable agentic behavior?")

    # Test 3: Simple question (should converge quickly)
    print("\n\n🧪 TEST 3: Simple Question (Quick Convergence)")
    response3 = self_rag(
        "What is reranking?", max_iterations=3, quality_threshold=0.8, verbose=True
    )
    print_self_rag_response(response3)
