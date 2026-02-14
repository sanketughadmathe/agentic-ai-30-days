import os

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
# 1. Sample docs
# -----------------------------
documents = [
    Document(page_content="ReAct combines reasoning and acting in iterative loops."),
    Document(page_content="Reranking improves retrieval precision."),
    Document(page_content="Structured outputs enforce deterministic contracts."),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
chunks = splitter.split_documents(documents)

# Use free embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatOpenAI(
    model="arcee-ai/trinity-large-preview:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# -----------------------------
# 2. Pydantic Models
# -----------------------------
class Answer(BaseModel):
    """Generated answer with metadata."""

    answer: str = Field(description="The actual answer to the question")
    used_context: bool = Field(
        description="Whether the answer uses only the provided context"
    )
    specific_claims: list[str] = Field(
        description="Key claims made in the answer", max_length=5
    )


class GroundingCheck(BaseModel):
    """Grounding verification result."""

    is_grounded: bool = Field(
        description="Whether answer is fully supported by context"
    )
    unsupported_claims: list[str] = Field(
        description="Claims not found in context (empty if grounded)", default=[]
    )
    reasoning: str = Field(description="Explanation of grounding assessment")


class ConfidenceAssessment(BaseModel):
    """Confidence scoring with breakdown."""

    overall_score: float = Field(
        description="Overall confidence 0.0-1.0", ge=0.0, le=1.0
    )
    context_coverage: float = Field(
        description="How well context covers the question", ge=0.0, le=1.0
    )
    answer_completeness: float = Field(
        description="How complete the answer is", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Explanation of confidence score")


class GuardedResponse(BaseModel):
    """Final response with all safety checks."""

    status: str = Field(description="success, fallback, or error")
    answer: str | None = Field(description="Answer if successful")
    confidence: float | None = Field(description="Confidence score if successful")
    message: str | None = Field(description="Explanation or error message")
    grounding_check: GroundingCheck | None = Field(description="Grounding details")
    retrieved_docs: list[str] = Field(description="Documents used")


# -----------------------------
# 3. Retrieve
# -----------------------------
def retrieve(question: str, k: int = 2):
    """Retrieve relevant documents."""
    return vectorstore.similarity_search(question, k=k)


# -----------------------------
# 4. Generate answer with structure
# -----------------------------
def generate(question: str, docs: list[Document]) -> Answer:
    """Generate answer with structured output."""

    llm_structured = llm.with_structured_output(Answer)

    context = "\n".join([f"- {d.page_content}" for d in docs])

    prompt = f"""Answer the question using ONLY the context provided below.

        Context:
        {context}

        Question: {question}

        Requirements:
        1. Use only information from the context
        2. Be specific and concise
        3. If the context is insufficient, state that clearly
        4. List the key claims you're making
    """

    return llm_structured.invoke(prompt)


# -----------------------------
# 5. Grounding check with details
# -----------------------------
def check_grounding(
    question: str, answer: Answer, docs: list[Document]
) -> GroundingCheck:
    """Verify answer is grounded in context."""

    llm_structured = llm.with_structured_output(GroundingCheck)

    context = "\n".join([f"- {d.page_content}" for d in docs])

    prompt = f"""Verify if the answer is fully supported by the context.

        Question: {question}

        Answer: {answer.answer}

        Key Claims in Answer:
        {chr(10).join([f"- {claim}" for claim in answer.specific_claims])}

        Context:
        {context}

        For each claim in the answer, check if it's supported by the context.
        List any claims that are NOT supported.
        The answer is only grounded if ALL claims are supported.
    """

    return llm_structured.invoke(prompt)


# -----------------------------
# 6. Confidence score with breakdown
# -----------------------------
def confidence_score(
    question: str, answer: Answer, docs: list[Document]
) -> ConfidenceAssessment:
    """Calculate confidence with detailed breakdown."""

    llm_structured = llm.with_structured_output(ConfidenceAssessment)

    context = "\n".join([f"- {d.page_content}" for d in docs])

    prompt = f"""Assess confidence in this answer on a scale of 0.0 to 1.0.

        Question: {question}

        Answer: {answer.answer}

        Context:
        {context}

        Provide three scores (each 0.0-1.0):
        1. Context coverage: How well does the context address the question?
        2. Answer completeness: How complete is the answer?
        3. Overall score: Combined confidence (average of above)

        Consider:
        - Is the context directly relevant?
        - Does the answer fully address the question?
        - Are there any gaps or uncertainties?
    """

    return llm_structured.invoke(prompt)


# -----------------------------
# 7. Guarded RAG pipeline
# -----------------------------
def guarded_rag(question: str, confidence_threshold: float = 0.5) -> GuardedResponse:
    """
    RAG pipeline with safety guardrails.

    Args:
        question: User's question
        confidence_threshold: Minimum confidence to return answer (0.0-1.0)
    """
    try:
        # Step 1: Retrieve
        docs = retrieve(question)
        doc_contents = [d.page_content for d in docs]

        # Step 2: Generate answer
        answer = generate(question, docs)

        # Step 3: Check grounding
        grounding = check_grounding(question, answer, docs)

        # Step 4: Calculate confidence
        confidence = confidence_score(question, answer, docs)

        # Step 5: Apply guardrails
        if not grounding.is_grounded:
            return GuardedResponse(
                status="fallback",
                answer=None,
                confidence=None,
                message=f"Answer contains unsupported claims: {', '.join(grounding.unsupported_claims)}",
                grounding_check=grounding,
                retrieved_docs=doc_contents,
            )

        if confidence.overall_score < confidence_threshold:
            return GuardedResponse(
                status="fallback",
                answer=None,
                confidence=confidence.overall_score,
                message=f"Confidence too low ({confidence.overall_score:.2f} < {confidence_threshold}). {confidence.reasoning}",
                grounding_check=grounding,
                retrieved_docs=doc_contents,
            )

        # Success!
        return GuardedResponse(
            status="success",
            answer=answer.answer,
            confidence=confidence.overall_score,
            message=f"Answer generated with {confidence.overall_score:.2f} confidence",
            grounding_check=grounding,
            retrieved_docs=doc_contents,
        )

    except Exception as e:
        return GuardedResponse(
            status="error",
            answer=None,
            confidence=None,
            message=f"Error: {str(e)}",
            grounding_check=None,
            retrieved_docs=[],
        )


# -----------------------------
# 8. Pretty print results
# -----------------------------
def print_guarded_result(question: str, result: GuardedResponse):
    """Format and print the guarded RAG result."""

    print("\n" + "=" * 70)
    print(f"❓ Question: {question}")
    print("=" * 70)

    print(f"\n🎯 Status: {result.status.upper()}")

    if result.retrieved_docs:
        print("\n📚 Retrieved Documents:")
        for i, doc in enumerate(result.retrieved_docs, 1):
            print(f"  {i}. {doc}")

    if result.grounding_check:
        print("\n🔍 Grounding Check:")
        print(
            f"  Grounded: {'✓ Yes' if result.grounding_check.is_grounded else '✗ No'}"
        )
        print(f"  Reasoning: {result.grounding_check.reasoning}")
        if result.grounding_check.unsupported_claims:
            print("  Unsupported Claims:")
            for claim in result.grounding_check.unsupported_claims:
                print(f"    - {claim}")

    if result.confidence is not None:
        print(f"\n📊 Confidence: {result.confidence:.2f}")

    if result.status == "success" and result.answer:
        print("\n✅ Answer:")
        print(f"  {result.answer}")
    else:
        print("\n⚠️  Message:")
        print(f"  {result.message}")

    print("=" * 70)


# -----------------------------
# 9. Run with multiple test cases
# -----------------------------
if __name__ == "__main__":
    # Test 1: Good question (should succeed)
    print("\n🧪 TEST 1: Well-covered question")
    question1 = "What is ReAct?"
    result1 = guarded_rag(question1)
    print_guarded_result(question1, result1)

    # Test 2: Question requiring more context (might fail)
    print("\n🧪 TEST 2: Question with insufficient context")
    question2 = "Explain ReAct and its benefits in detail."
    result2 = guarded_rag(question2)
    print_guarded_result(question2, result2)

    # Test 3: Off-topic question (should fail)
    print("\n🧪 TEST 3: Off-topic question")
    question3 = "How do I make pizza?"
    result3 = guarded_rag(question3)
    print_guarded_result(question3, result3)

    # Access structured data
    print("\n📊 Programmatic Access Example:")
    print(f"Status: {result1.status}")
    print(f"Confidence: {result1.confidence}")
    print(
        f"Is Grounded: {result1.grounding_check.is_grounded if result1.grounding_check else 'N/A'}"
    )
    """
    output
        🧪 TEST 1: Well-covered question

        ======================================================================
        ❓ Question: What is ReAct?
        ======================================================================

        🎯 Status: SUCCESS

        📚 Retrieved Documents:
        1. ReAct combines reasoning and acting in iterative loops.
        2. Structured outputs enforce deterministic contracts.

        🔍 Grounding Check:
        Grounded: ✓ Yes
        Reasoning: Both claims in the answer are directly supported by the context. The first claim about ReAct combining reasoning and acting in iterative loops is explicitly stated in the context. The second claim about structured outputs enforcing deterministic contracts is also directly mentioned in the context. Since all claims are supported, the answer is fully grounded.

        📊 Confidence: 0.80

        ✅ Answer:
        ReAct is a framework that combines reasoning and acting in iterative loops, with structured outputs enforcing deterministic contracts.
        ======================================================================

        🧪 TEST 2: Question with insufficient context

        ======================================================================
        ❓ Question: Explain ReAct and its benefits in detail.
        ======================================================================

        🎯 Status: FALLBACK

        📚 Retrieved Documents:
        1. ReAct combines reasoning and acting in iterative loops.
        2. Reranking improves retrieval precision.

        🔍 Grounding Check:
        Grounded: ✗ No
        Reasoning: The claim that ReAct combines reasoning and acting in iterative loops is supported by the context. However, the claim that the benefit of ReAct is that it integrates reasoning and acting processes, allowing for more dynamic and adaptive decision-making, is not supported by the context. The context only mentions that ReAct combines reasoning and acting in iterative loops, but it does not provide any information about the benefits of this integration.
        Unsupported Claims:
            - The benefit of ReAct is that it integrates reasoning and acting processes, allowing for more dynamic and adaptive decision-making.

        ⚠️  Message:
        Answer contains unsupported claims: The benefit of ReAct is that it integrates reasoning and acting processes, allowing for more dynamic and adaptive decision-making.
        ======================================================================

        🧪 TEST 3: Off-topic question

        ======================================================================
        ❓ Question: How do I make pizza?
        ======================================================================

        🎯 Status: FALLBACK

        📚 Retrieved Documents:
        1. ReAct combines reasoning and acting in iterative loops.
        2. Structured outputs enforce deterministic contracts.

        🔍 Grounding Check:
        Grounded: ✗ No
        Reasoning: The answer makes two claims: (1) the context does not contain information on how to make pizza, and (2) the context only discusses ReAct and structured outputs, which are unrelated to pizza-making. Both claims are supported by the context, which explicitly mentions ReAct and structured outputs but does not mention pizza-making. Therefore, the answer is fully grounded.
        Unsupported Claims:
            - The context provided does not contain information on how to make pizza.
            - The context only discusses ReAct and structured outputs, which are unrelated to pizza-making.

        ⚠️  Message:
        Answer contains unsupported claims: The context provided does not contain information on how to make pizza., The context only discusses ReAct and structured outputs, which are unrelated to pizza-making.
        ======================================================================

        📊 Programmatic Access Example:
        Status: success
        Confidence: 0.8
        Is Grounded: True
    """
