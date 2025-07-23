from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang='en')
query = "The geopolitical history of India and Pakistan from the perspective of the Chinese."
docs = retriever.invoke(query)
print(docs)

# A retriever is determined by which data source it works with and what search strategy they use (MMR, Multi-Query, etc)
