from sentence_transformers import SentenceTransformer

transformer = SentenceTransformer("all-MiniLM-L6-v2")
def createEmbeddings(chunks:list[str]):
    embeddings = transformer.encode(chunks, show_progress_bar=True)
    return embeddings.tolist()