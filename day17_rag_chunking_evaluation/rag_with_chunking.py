# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# 1. Sample documents
# -----------------------------
documents = [
    Document(
        page_content="Agent memory allows state retention across reasoning steps."
    ),
    Document(page_content="ReAct combines reasoning and acting in a loop."),
    Document(page_content="Structured outputs enforce contracts in LLM systems."),
]


# -----------------------------
# 2. Chunking
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

chunks = text_splitter.split_documents(documents)


# -----------------------------
# 3. Embeddings + Vector store
# -----------------------------
# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# -----------------------------
# 4. Query
# -----------------------------
query = "What is ReAct?"

retrieved_docs = retriever.invoke(query)

print("\nRetrieved Chunks:\n")
for doc in retrieved_docs:
    print("-", doc.page_content)
