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
docs = [
    Document(page_content="ReAct combines reasoning and acting in iterative loops."),
    Document(page_content="Agent memory stores intermediate reasoning steps."),
    Document(page_content="Structured outputs enforce deterministic contracts."),
]

# -----------------------------
# 2. Chunk + vector store
# -----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# -----------------------------
# 3. LLM for query rewriting
# -----------------------------
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
)


# ---------------------------------------
# Pydantic Model for Structured Output
# ---------------------------------------
class QueryVariations(BaseModel):
    """Alternative search queries for better retrieval."""

    queries: list[str] = Field(
        description="3-5 alternative phrasings of the question",
        min_length=3,
        max_length=5,
    )
    reasoning: str = Field(description="Brief explanation of query strategy")


def generate_queries(question: str) -> QueryVariations:
    """Generate multiple query variations using structured output."""

    llm_structured = llm.with_structured_output(QueryVariations)

    prompt = f"""Generate alternative search queries to improve retrieval for this question.

        Question: {question}

        Create 3-5 variations that:
        1. Rephrase the question differently
        2. Break down complex questions into sub-questions
        3. Use different terminology or synonyms
        4. Ask for specific aspects or components

        Each query should target different relevant documents.
    """

    result = llm_structured.invoke(prompt)
    return result


# ----------------------------------------
# 4. Multi-query retrieval with scoring
# ----------------------------------------
def multi_query_retrieve(question: str, k: int = 2, verbose: bool = True):
    """
    Retrieve using multiple query variations.

    Args:
        question: Original user question
        k: Number of documents to retrieve per query
        verbose: Print query generation details
    """
    # Generate query variations
    query_variations = generate_queries(question)

    if verbose:
        print("\n" + "=" * 60)
        print("🔍 Generated Query Variations:")
        print("=" * 60)
        print(f"Strategy: {query_variations.reasoning}\n")
        for i, q in enumerate(query_variations.queries, 1):
            print(f"{i}. {q}")
        print("=" * 60)

    # Retrieve for each query
    all_docs = []
    doc_scores = {}  # Track how many times each doc appears

    for query in query_variations.queries:
        results = vectorstore.similarity_search(query, k=k)

        for doc in results:
            content = doc.page_content
            all_docs.append(doc)
            doc_scores[content] = doc_scores.get(content, 0) + 1

    # Deduplicate and sort by frequency (more appearances = more relevant)
    unique_docs = {doc.page_content: doc for doc in all_docs}
    sorted_docs = sorted(
        unique_docs.values(), key=lambda d: doc_scores[d.page_content], reverse=True
    )

    if verbose:
        print("\n📊 Document Relevance Scores:")
        print("=" * 60)
        for doc in sorted_docs:
            score = doc_scores[doc.page_content]
            print(f"[{score}x] {doc.page_content}")
        print("=" * 60)

    return sorted_docs, query_variations


# -----------------------------
# 5. Evaluation
# -----------------------------
class RetrievalEvaluation(BaseModel):
    """Evaluation of multi-query retrieval quality."""

    overall_score: str = Field(description="HIGH, MEDIUM, or LOW")
    coverage: str = Field(description="How well queries covered different aspects")
    redundancy: str = Field(description="Level of duplicate/overlapping results")
    answerable: bool = Field(description="Can the question be answered?")
    explanation: str = Field(description="Brief assessment")


def evaluate_retrieval(question: str, docs: list[Document], queries: QueryVariations):
    """Evaluate the quality of multi-query retrieval."""

    llm_eval = llm.with_structured_output(RetrievalEvaluation)

    formatted_docs = "\n".join([f"- {d.page_content}" for d in docs])
    formatted_queries = "\n".join(
        [f"{i + 1}. {q}" for i, q in enumerate(queries.queries)]
    )

    prompt = f"""Evaluate this multi-query retrieval result.

        Original Question: {question}

        Generated Queries:
        {formatted_queries}

        Retrieved Documents:
        {formatted_docs}

        Assess:
        - Overall relevance (HIGH/MEDIUM/LOW)
        - Query coverage (did different queries find different aspects?)
        - Redundancy (too much overlap vs good diversity)
        - Whether the question can be answered
    """

    return llm_eval.invoke(prompt)


# -----------------------------
# 6. Run with full pipeline
# -----------------------------
if __name__ == "__main__":
    question = "How does ReAct work?"

    print(f"\n❓ Question: {question}\n")

    # Multi-query retrieval
    retrieved_docs, queries = multi_query_retrieve(question, k=2, verbose=True)

    # Evaluate
    print("\n🎯 Retrieval Evaluation:")
    print("=" * 60)
    evaluation = evaluate_retrieval(question, retrieved_docs, queries)
    print(f"Score: {evaluation.overall_score}")
    print(f"Coverage: {evaluation.coverage}")
    print(f"Redundancy: {evaluation.redundancy}")
    print(f"Answerable: {evaluation.answerable}")
    print(f"\n{evaluation.explanation}")
    print("=" * 60)

    # Final context
    print("\n📄 Final Retrieved Context:")
    print("=" * 60)
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc.page_content}")
    print("=" * 60)
    print("=" * 60)
