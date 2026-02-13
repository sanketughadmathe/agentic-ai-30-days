import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

# Fix tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# 1. Sample documents
# -----------------------------
documents = [
    Document(page_content="ReAct combines reasoning and acting in iterative loops."),
    Document(page_content="Agent memory stores intermediate reasoning steps."),
    Document(page_content="Structured outputs enforce deterministic contracts."),
    Document(page_content="Reranking improves retrieval precision."),
]

# -----------------------------
# 2. Chunking + Vector Store
# -----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# -----------------------------
# 3. LLM Setup (Gemini)
# -----------------------------
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
)


# -------------------------------------------
# 4. Pydantic Models for Structured Output
# -------------------------------------------
class DocumentScore(BaseModel):
    """Relevance score for a single document."""

    score: float = Field(
        description="Relevance score from 0.0 to 10.0", ge=0.0, le=10.0
    )
    reasoning: str = Field(description="Brief explanation for the score")
    key_terms: list[str] = Field(
        description="Key terms that influenced the score", max_length=5
    )


class RerankingResult(BaseModel):
    """Complete reranking result for all documents."""

    document_scores: list[DocumentScore] = Field(
        description="Scores for each document in order"
    )
    top_k_indices: list[int] = Field(description="Indices of top documents (0-based)")


# -----------------------------
# 5. Retrieve candidates
# -----------------------------
def retrieve(question: str, k: int = 4):
    """Retrieve initial candidates using vector similarity."""
    return vectorstore.similarity_search(question, k=k)


# ----------------------------------------
# 6. Rerank with LLM (Batch Processing)
# ----------------------------------------
def rerank_batch(question: str, docs: list[Document], top_k: int = 2) -> tuple:
    """
    Rerank documents using LLM with structured output (batch mode).

    Args:
        question: User's question
        docs: Candidate documents to rerank
        top_k: Number of top documents to return

    Returns:
        Tuple of (reranked_docs, reranking_result)
    """
    llm_structured = llm.with_structured_output(RerankingResult)

    # Format all documents for batch evaluation
    formatted_docs = "\n\n".join(
        [f"Document {i}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""Score each document's relevance for answering the question.

        Question: {question}

        Documents:
        {formatted_docs}

        For each document, provide:
        1. A relevance score (0.0 = irrelevant, 10.0 = perfectly relevant)
        2. Brief reasoning for the score
        3. Key terms that influenced the score

        Then identify the indices of the top {top_k} most relevant documents.
    """

    result = llm_structured.invoke(prompt)

    # Reorder documents based on scores
    doc_score_pairs = [(docs[i], result.document_scores[i]) for i in range(len(docs))]
    doc_score_pairs.sort(key=lambda x: x[1].score, reverse=True)

    reranked_docs = [doc for doc, _ in doc_score_pairs]

    return reranked_docs, result


def rerank_sequential(question: str, docs: list[Document]) -> list[Document]:
    """
    Rerank documents one-by-one (alternative approach).
    Slower but works if batch reranking has issues.
    """
    llm_structured = llm.with_structured_output(DocumentScore)

    scored_docs = []

    for doc in docs:
        prompt = f"""Score this document's relevance for answering the question.

        Question: {question}

        Document: {doc.page_content}

        Provide a score from 0.0 (irrelevant) to 10.0 (perfectly relevant).
    """

        score_result = llm_structured.invoke(prompt)
        scored_docs.append((score_result.score, doc, score_result))

    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc, _ in scored_docs]


# -----------------------------
# 7. Generate final answer
# -----------------------------
class FinalAnswer(BaseModel):
    """Structured answer with context."""

    answer: str = Field(description="Direct answer to the question")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW confidence")
    reasoning: str = Field(description="How the answer was derived from context")


def generate_answer(question: str, docs: list[Document]) -> FinalAnswer:
    """Generate answer with structured output."""

    llm_structured = llm.with_structured_output(FinalAnswer)

    context = "\n".join([f"- {doc.page_content}" for doc in docs])

    prompt = f"""Answer the question using ONLY the context provided.

        Context:
        {context}

        Question: {question}

        Provide:
        1. A direct answer
        2. Your confidence level (HIGH/MEDIUM/LOW)
        3. Brief explanation of how you derived the answer from the context

        If the context doesn't contain enough information, state that clearly.
    """

    return llm_structured.invoke(prompt)


# -----------------------------
# 8. Pretty Print Results
# -----------------------------
def print_results(
    question: str,
    candidates: list[Document],
    reranked: list[Document],
    reranking_result: RerankingResult,
    top_docs: list[Document],
    answer: FinalAnswer,
):
    """Print formatted results."""

    print("\n" + "=" * 70)
    print(f"❓ Question: {question}")
    print("=" * 70)

    print("\n📋 Initial Candidates (Vector Similarity):")
    print("-" * 70)
    for i, doc in enumerate(candidates, 1):
        print(f"{i}. {doc.page_content}")

    print("\n🎯 Reranking Scores:")
    print("-" * 70)
    for i, (doc, score_info) in enumerate(
        zip(reranked, reranking_result.document_scores)
    ):
        print(f"\n{i + 1}. Score: {score_info.score:.1f}/10.0")
        print(f"   Content: {doc.page_content}")
        print(f"   Reasoning: {score_info.reasoning}")
        print(f"   Key Terms: {', '.join(score_info.key_terms)}")

    print("\n✨ Selected Top Documents:")
    print("-" * 70)
    for i, doc in enumerate(top_docs, 1):
        print(f"{i}. {doc.page_content}")

    print("\n💡 Final Answer:")
    print("-" * 70)
    print(f"Answer: {answer.answer}")
    print(f"Confidence: {answer.confidence}")
    print(f"Reasoning: {answer.reasoning}")
    print("=" * 70)


# -----------------------------
# 9. Run Pipeline
# -----------------------------
if __name__ == "__main__":
    question = "How does ReAct work?"

    # Step 1: Initial retrieval
    candidates = retrieve(question, k=4)

    # Step 2: Rerank with LLM
    reranked, reranking_result = rerank_batch(question, candidates, top_k=2)

    # Step 3: Select top documents
    top_docs = reranked[:2]

    # Step 4: Generate answer
    answer = generate_answer(question, top_docs)

    # Step 5: Display results
    print_results(question, candidates, reranked, reranking_result, top_docs, answer)

    # Alternative: Test sequential reranking
    print("\n\n🔄 Testing Sequential Reranking (Alternative Method):")
    print("=" * 70)
    reranked_seq = rerank_sequential(question, candidates)
    print("\nReranked Order (Sequential):")
    for i, doc in enumerate(reranked_seq, 1):
        print(f"{i}. {doc.page_content}")
