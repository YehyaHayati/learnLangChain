from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

vector_store = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings())
retriever = vector_store.as_retriever(search_kwargs={'k':2})
query = "What are embeddings used for?"
results = retriever.invoke(query)
print([result.page_content for result in results])