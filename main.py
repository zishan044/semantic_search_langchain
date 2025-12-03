
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

def main():

    file_path = "example_data/nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    print(len(all_splits))

    embeddings = OllamaEmbeddings(model="llama3.1")

    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    print(vector_1[:10])



if __name__ == "__main__":
    main()
