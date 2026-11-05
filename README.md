# HR Policy Chatbot using RAG

## Overview

This project implements an AI-powered HR Policy Chatbot using Retrieval-Augmented Generation (RAG).

The application allows users to ask questions about company HR policies and receive context-aware answers based on the information retrieved from an HR policy document.

It uses ChromaDB for vector storage and semantic retrieval, along with the OpenAI API for generating responses.

The project also demonstrates conversational memory, allowing the chatbot to use previous questions and responses to understand follow-up queries.

## Features

* **Document Loading:** Loads HR policy information from a text document.
* **Document Chunking:** Splits the HR policy document into smaller sections based on section headings.
* **Vector Database:** Stores document chunks in ChromaDB for semantic retrieval.
* **Semantic Search:** Retrieves relevant document sections based on user queries.
* **Context Building:** Combines retrieved sections to provide context for answer generation.
* **AI-Powered Responses:** Uses the OpenAI API to generate answers based on the retrieved HR policy information.
* **Conversational Memory:** Maintains previous user questions and chatbot responses to support follow-up queries.
* **Policy-Based Responses:** Instructs the chatbot to respond only using the provided HR policy context and indicate when information is unavailable.

## Implementation Approaches

The notebook demonstrates three implementation stages:

### 1. Basic RAG Pipeline

Implemented using:

* ChromaDB
* OpenAI API

The pipeline loads the HR document, splits it into sections, stores the chunks in a vector database, retrieves relevant information, and generates answers using the OpenAI API.

### 2. RAG with Conversational Memory

Extended the basic RAG pipeline by maintaining a chat history containing previous user questions and assistant responses.

The conversation history is passed along with the retrieved HR policy context to help the chatbot understand follow-up questions.

### 3. RAG using LangChain

Demonstrates an alternative implementation using the LangChain framework.

Components include:

* LangChain Document
* ChatPromptTemplate
* OpenAIEmbeddings
* ChatOpenAI
* Chroma Vector Store
* StrOutputParser

## Technologies Used

* Python
* OpenAI API
* ChromaDB
* LangChain
* LangChain OpenAI
* LangChain Chroma
* Pandas
* Jupyter Notebook

## RAG Workflow

1. Load the HR policy document.
2. Split the document into individual sections.
3. Convert document chunks into vector embeddings.
4. Store the embeddings in ChromaDB.
5. Accept a question from the user.
6. Retrieve relevant document chunks using semantic similarity search.
7. Combine retrieved information into context.
8. Include conversation history when available.
9. Pass the context and question to the language model.
10. Generate a context-aware answer.

## Conversational Memory

The chatbot maintains a conversation history using a Python list containing user and assistant messages.

This history is passed to the language model along with the retrieved document context, helping it understand follow-up questions.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hr-policy-rag-chatbot.git
```

### 2. Navigate to the project folder

```bash
cd hr-policy-rag-chatbot
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the OpenAI API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_actual_api_key
```

Never commit this file to GitHub.

### 7. Run the notebook

Open the notebook using Jupyter Notebook or VS Code and execute the cells in order.

Update the HR document path and API key configuration according to your local environment.

## Project Output

The chatbot accepts HR-related questions, retrieves relevant sections from the policy document, and generates answers using the OpenAI API.

It also maintains conversation history to support follow-up questions.

## Author

Bhavana Bhageerathi
