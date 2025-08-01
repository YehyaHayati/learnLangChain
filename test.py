from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )
docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='my_chroma_db',
    collection_name='sample'    # A collection is basically a table in which embeddings and metadata is stored
)
vector_store.add_documents(docs)
print(vector_store.get(include=['embeddings', 'documents', 'metadatas']))

print(vector_store.similarity_search_with_score(query='Who among these are a bowler?', k=2))
# Where k is the k-nearest vectors

print(vector_store.similarity_search_with_score(query="", filter={"team": "Chennai Super Kings"}))
# Filtering by metadata

# vector_store.delete(ids=['5daf958a-5136-4ddd-9fb3-26c1f43fa455', 'bf0b8615-3b5b-430c-ac2c-cd94614dd65d', '1086d84d-d3cd-4c34-8082-47b3d25fd532', '938f35d2-a4b5-4bf1-b34e-2fcf53188973', 'dc39cff1-46d2-41f5-871b-b5dc07cda863'])
print(vector_store.get(include=['embeddings', 'documents', 'metadatas']))

updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

# vector_store.update_document(document_id='c8a87895-3848-4249-a898-c910ebcd6777', document=updated_doc1)
print(vector_store.get(include=['embeddings', 'documents', 'metadatas']))