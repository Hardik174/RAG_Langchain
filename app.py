from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
from src.embedding import EmbeddingPipeline

# Example usage
if __name__ == "__main__":
    
    # docs = load_all_documents("data")
    # chunks = EmbeddingPipeline().chunk_documents(docs)
    # chunk_vectors = EmbeddingPipeline().embed_chunks(chunks)
    # print(chunk_vectors)
    store = FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    store.load()
    # print(store.query("Explain Transformer Architecture", top_k=3))
    rag_search = RAGSearch()
    query = "Explain Transformer Architecture"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)