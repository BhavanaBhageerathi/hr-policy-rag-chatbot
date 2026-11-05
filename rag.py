"""
HR Policy RAG Chatbot
---------------------
A command-line HR policy chatbot with:
1. A ChromaDB + OpenAI implementation.
2. An alternative LangChain + Chroma + OpenAI implementation.
3. Conversational memory for follow-up questions.

Setup:
    pip install -r requirements.txt

Create a .env file in the project root:
    OPENAI_API_KEY=your_api_key
    OPENAI_MODEL=gpt-4.1-mini
    IMPLEMENTATION=native

Place your non-confidential sample policy document at:
    documents/companyhrdoc.txt

Run:
    python hr_policy_rag_chatbot.py

Type 'exit' to stop the chatbot.
"""

import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- CONFIGURATION --------------------

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_PATH = BASE_DIR / "documents" / "companyhrdoc.txt"
CHROMA_PATH = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
IMPLEMENTATION = os.getenv("IMPLEMENTATION", "native").strip().lower()

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing. Add it to a .env file in the project root."
    )

client = OpenAI(api_key=API_KEY)


# -------------------- DOCUMENT PROCESSING --------------------

def load_document(file_path):
    """Load the HR policy document from a text file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Policy document not found: {file_path}\n"
            "Place companyhrdoc.txt inside the documents folder."
        )

    with open(file_path, "r", encoding="utf-8") as file:
        document = file.read()

    if not document.strip():
        raise ValueError("The HR policy document is empty.")

    return document


def chunk_document(document):
    """
    Split the document at headings such as 'Section 1:'.
    If no such headings exist, use the complete document as one chunk.
    """
    sections = re.split(r"(?=Section\s+\d+\s*:)", document, flags=re.IGNORECASE)
    chunks = [section.strip() for section in sections if section.strip()]

    return chunks if chunks else [document.strip()]


# -------------------- NATIVE CHROMADB IMPLEMENTATION --------------------

def create_vector_database(chunks):
    """Create or update a persistent ChromaDB collection."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collection = chroma_client.get_or_create_collection(
        name="hr_policy_native"
    )

    # Upsert avoids duplicate-ID errors when the script is run again.
    ids = [f"policy_chunk_{index}" for index in range(len(chunks))]
    collection.upsert(ids=ids, documents=chunks)

    return collection


def retrieve_relevant_chunks(query, collection, top_k=3):
    """Retrieve the most relevant policy chunks for a question."""
    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, total_chunks)
    )

    if not results.get("documents") or not results["documents"][0]:
        return []

    return results["documents"][0]


def build_context(documents):
    """Combine retrieved chunks into one context string."""
    return "\n\n".join(documents)


# -------------------- LANGCHAIN IMPLEMENTATION --------------------

def create_langchain_vector_database(chunks):
    """Create a persistent Chroma vector store using LangChain."""
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    documents = [
        Document(page_content=chunk, metadata={"chunk": index})
        for index, chunk in enumerate(chunks)
    ]

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="hr_policy_langchain",
        persist_directory=str(CHROMA_PATH)
    )

    return vector_store


def retrieve_langchain_chunks(query, vector_store, top_k=3):
    """Retrieve relevant Document objects from the LangChain vector store."""
    return vector_store.similarity_search(query, k=top_k)


def build_langchain_context(documents):
    """Build context from LangChain Document objects."""
    return "\n\n".join(document.page_content for document in documents)


# -------------------- ANSWER GENERATION + MEMORY --------------------

def format_chat_history(chat_history):
    """Convert chat history into readable text for the language model."""
    if not chat_history:
        return "No previous conversation."

    return "\n".join(
        f"{item['role']}: {item['content']}"
        for item in chat_history[-10:]
    )


def generate_answer(query, context, chat_history):
    """Generate an answer using retrieved policy context and chat history."""
    history_text = format_chat_history(chat_history)

    prompt = f"""
You are an HR policy assistant.

Answer the user's question using only the information in the HR policy
context below. Do not invent or assume policy details.

Use the conversation history only to understand references and follow-up
questions. The policy context is the source of truth.

If the answer is not supported by the policy context, respond:
"Sorry, it is not mentioned in the policy."

HR POLICY CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

CURRENT USER QUESTION:
{query}

Provide a clear and concise answer.
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt
    )

    return response.output_text.strip()


def make_retrieval_query(user_query, chat_history):
    """
    Add recent conversation context to short follow-up questions so retrieval
    has a better chance of finding the correct policy section.
    """
    if not chat_history:
        return user_query

    recent_history = format_chat_history(chat_history[-4:])
    return (
        f"Previous conversation:\n{recent_history}\n\n"
        f"Current question: {user_query}"
    )


# -------------------- CHATBOT PIPELINES --------------------

def run_native_pipeline(chunks):
    """Run the RAG chatbot using ChromaDB directly."""
    collection = create_vector_database(chunks)
    print("\nHR policy document indexed in ChromaDB.")

    chat_history = []

    while True:
        user_query = input("\nWhat would you like to ask HR? ").strip()

        if user_query.lower() in {"exit", "quit"}:
            print("Chatbot session ended.")
            break

        if not user_query:
            print("Please enter a question.")
            continue

        retrieval_query = make_retrieval_query(user_query, chat_history)
        documents = retrieve_relevant_chunks(
            retrieval_query, collection, top_k=3
        )

        if not documents:
            answer = "Sorry, it is not mentioned in the policy."
        else:
            context = build_context(documents)
            answer = generate_answer(user_query, context, chat_history)

        print("\nAnswer:", answer)

        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": answer})


def run_langchain_pipeline(chunks):
    """Run the RAG chatbot using LangChain and Chroma."""
    vector_store = create_langchain_vector_database(chunks)
    print("\nHR policy document indexed using LangChain and Chroma.")

    chat_history = []

    while True:
        user_query = input("\nWhat would you like to ask HR? ").strip()

        if user_query.lower() in {"exit", "quit"}:
            print("Chatbot session ended.")
            break

        if not user_query:
            print("Please enter a question.")
            continue

        retrieval_query = make_retrieval_query(user_query, chat_history)
        documents = retrieve_langchain_chunks(
            retrieval_query, vector_store, top_k=3
        )

        if not documents:
            answer = "Sorry, it is not mentioned in the policy."
        else:
            context = build_langchain_context(documents)
            answer = generate_answer(user_query, context, chat_history)

        print("\nAnswer:", answer)

        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": answer})


# -------------------- MAIN --------------------

def main():
    print("HR POLICY QUESTION ANSWERING CHATBOT")
    print("------------------------------------")

    document = load_document(DOCUMENT_PATH)
    print("HR policy document loaded successfully.")

    chunks = chunk_document(document)
    print(f"Document split into {len(chunks)} chunks.")

    if IMPLEMENTATION == "native":
        run_native_pipeline(chunks)
    elif IMPLEMENTATION == "langchain":
        run_langchain_pipeline(chunks)
    else:
        raise ValueError(
            "Invalid IMPLEMENTATION value. Choose 'native' or 'langchain' "
            "in your .env file."
        )


if __name__ == "__main__":
    main()
