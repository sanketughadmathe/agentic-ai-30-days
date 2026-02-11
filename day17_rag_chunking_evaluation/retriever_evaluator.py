import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0,
)


# -----------------------------
# Pydantic Models
# -----------------------------
class DocumentRelevance(BaseModel):
    """Relevance score for a single document."""

    document_index: int = Field(description="Index of the document (0-based)")
    score: str = Field(description="HIGH, MEDIUM, or LOW")
    reasoning: str = Field(description="Brief explanation for this score")


class RelevanceEvaluation(BaseModel):
    """Complete evaluation of retrieved documents."""

    overall_score: str = Field(description="Overall relevance: HIGH, MEDIUM, or LOW")
    explanation: str = Field(description="Brief explanation of overall assessment")
    document_scores: list[DocumentRelevance] = Field(
        description="Individual scores for each document"
    )
    answerable: bool = Field(
        description="Can the question be answered with these documents?"
    )


def evaluate_relevance(question: str, docs: list[str]) -> dict:
    """
    Evaluates retrieval quality using an LLM judge with structured output.

    Args:
        question: The user's question
        docs: List of retrieved document chunks

    Returns:
        RelevanceEvaluation object with scores and explanations

    """
    # Create LLM with structured output
    llm_structured = llm.with_structured_output(RelevanceEvaluation)

    # Format documents with numbering
    formatted_docs = "\n".join([f"{i}. {doc}" for i, doc in enumerate(docs)])

    prompt = f"""You are evaluating the quality of retrieved documents for a RAG system.

        Question: {question}

        Retrieved Documents:
        {formatted_docs}

        Evaluate each document's relevance:
        - HIGH: Document directly answers or strongly supports answering the question
        - MEDIUM: Document is somewhat related but incomplete or tangential
        - LOW: Document is irrelevant or unrelated

        Provide:
        1. An overall score (HIGH/MEDIUM/LOW) based on whether these docs can answer the question
        2. Individual scores for each document with reasoning
        3. Whether the question is answerable with these documents
    """

    result = llm_structured.invoke(prompt)
    return result


def print_evaluation(evaluation: RelevanceEvaluation, question: str, docs: list[str]):
    """Pretty print the evaluation results."""
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    print(f"\n📊 Overall Score: {evaluation.overall_score}")
    print(f"💡 Explanation: {evaluation.explanation}")
    print(f"✓ Answerable: {'Yes' if evaluation.answerable else 'No'}")

    print("\n📄 Document Scores:")
    for doc_score in evaluation.document_scores:
        print(f"\n  Doc {doc_score.document_index}: {doc_score.score}")
        print(f"  Content: {docs[doc_score.document_index]}")
        print(f"  Reasoning: {doc_score.reasoning}")
    print("=" * 60)


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Test Case 1: Good retrieval
    print("\n🧪 TEST 1: Good Retrieval")
    question1 = "What is ReAct?"
    docs1 = [
        "ReAct combines reasoning and acting in a loop.",
        "Agent memory allows state retention across reasoning steps.",
    ]

    eval1 = evaluate_relevance(question1, docs1)
    print_evaluation(eval1, question1, docs1)

    # Test Case 2: Poor retrieval
    print("\n🧪 TEST 2: Poor Retrieval")
    question2 = "What is ReAct?"
    docs2 = [
        "Python is a programming language.",
        "The sky is blue on a clear day.",
    ]

    eval2 = evaluate_relevance(question2, docs2)
    print_evaluation(eval2, question2, docs2)

    # Test Case 3: Mixed retrieval
    print("\n🧪 TEST 3: Mixed Retrieval")
    question3 = "How does agent memory work?"
    docs3 = [
        "Agent memory allows state retention across reasoning steps.",
        "ReAct combines reasoning and acting in a loop.",
        "Python uses garbage collection for memory management.",
    ]

    eval3 = evaluate_relevance(question3, docs3)
    print_evaluation(eval3, question3, docs3)

    # Access structured data programmatically
    print("\n📈 Programmatic Access Example:")
    print(f"Overall Score: {eval1.overall_score}")
    print(f"Is Answerable: {eval1.answerable}")
    print(f"First Doc Score: {eval1.document_scores[0].score}")
