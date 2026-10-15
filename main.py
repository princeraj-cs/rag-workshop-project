from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    api_key=API_KEY,
    model="openai/gpt-oss-120b",
)

# 1. Load the documents from retrieval 
text_loader = TextLoader("technova_policy.txt")
documents = text_loader.load()


# 2. Split the documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
)

chunks = splitter.split_documents(documents)


# 3. Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectors = embedding_model.embed_documents([doc.page_content for doc in chunks])

# 4. Store embeddings in Vector DB
vector_store = FAISS.from_documents(chunks, embedding_model)

# 5. Create retriver
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 6. Create RAG Chain

def format_docs(docs):
    "Return the single steing representation of the list of documents."
    return "\n\n".join([f"Document {i+1}:\n {doc.page_content}" for i, doc in enumerate(docs)])


template = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Use three sentences maximum to answer the question.

Context: {context}

Question: {question}

Answer: 

"""

prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


query_1 = "How many paid annual leave days does a full-time employee receive per year?"

'''
Ans: A full‑time employee receives **18 paid annual leave days per year**. This figure is stated directly in the first document. No other source in the provided context gives a different amount.
'''

query_2 = "An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?"

"""
Ans: The policies permit remote work on Wednesdays (and Fridays) as long as the employee gives at least one working‑day notice.  
Leave on Friday is also allowed, but it must be requested at least one working day in advance.  
So both requests can be approved provided the advance‑notice requirement is met.
"""

query_3 = "What is the company's parental leave policy?"

"""
Ans: I don’t know the company’s parental leave policy.
"""

queries = [query_1, query_2, query_3]

for query in queries:
    response = rag_chain.invoke(query)
    print()
    print(f"Q. {query_1}")
    print(f"Answer from LLM: {response}")
    print()

