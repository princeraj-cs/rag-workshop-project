# RAG Workshop Project

This project is a simple Retrieval-Augmented Generation (RAG) demo built with Python, LangChain, FAISS, and Hugging Face embeddings. It loads a policy document, splits it into chunks, embeds those chunks, stores them in a vector database, and answers questions using a Groq-hosted LLM.

The example uses a fictional company policy document, `technova_policy.txt`, and demonstrates how a chatbot can answer questions based on the retrieved context instead of guessing from general knowledge.

## Features

- Text document ingestion using LangChain
- Chunking with `RecursiveCharacterTextSplitter`
- Embedding generation with `sentence-transformers/all-MiniLM-L6-v2`
- Vector search with FAISS
- Prompt-based question answering using Groq LLM
- Example business-policy queries included in `main.py`

## Project Structure

- `main.py` – main RAG pipeline and query execution
- `technova_policy.txt` – sample employee policy document used as the knowledge source
- `requirements.txt` – Python dependencies for the project
- `pyproject.toml` – project metadata and dependency configuration
- `.env` – environment variables (not committed by default; add your own)
- `README.md` – project documentation

## Tech Stack

- Python
- LangChain
- LangChain Groq integration
- FAISS
- Hugging Face sentence-transformers
- Groq LLM API

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

Or, if you are using the project configuration:

```bash
pip install -e .
```

## Environment Variables

This project expects a Groq API key in an environment variable named `GROQ_API_KEY`.

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Run the Project

```bash
python main.py
```

The script will:

- load the document from `technova_policy.txt`
- split it into text chunks
- create embeddings
- store them in a FAISS vector store
- run a set of example questions through the RAG pipeline

## Example Questions

The project includes example questions such as:

- How many paid annual leave days are given per year?
- Can an employee work remotely on Wednesday and take leave on Friday?
- What is the company's parental leave policy?

## Notes

- The sample knowledge base is intentionally limited to the policy file in this repository.
- The RAG system answers using the retrieved context, so its replies depend on the document content.
- If the answer is not present in the policy document, the LLM is prompted to say it does not know rather than hallucinate.

## License

This project does not currently declare a specific license. If you plan to share or distribute it publicly, consider adding a license file such as MIT or Apache-2.0.
