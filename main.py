import asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm.asyncio import tqdm_asyncio

async def embed_batch(documents, embeddings):
    """Embed a batch of documents asynchronously"""
    tasks = []
    for doc in documents:
        tasks.append(asyncio.to_thread(embeddings.embed_query, doc.page_content))
    return await asyncio.gather(*tasks)

async def main_async():

    loader = PyPDFLoader("example_data/nke-10k-2023.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model="llama3.1")
    vector_store = Chroma(
        collection_name="async_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_async_db",
    )
    
    batch_size = 5
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}")
        await embed_batch(batch, embeddings)
        vector_store.add_documents(documents=batch)
    
    results = vector_store.similarity_search_with_score("Nike revenue 2023", k=2)
    for doc, score in results:
        print(f"Score: {score}")
        print(doc.page_content[:150])

if __name__ == "__main__":
    asyncio.run(main_async())