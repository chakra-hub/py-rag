from transformers import AutoTokenizer
from docling.chunking import HybridChunker

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

chunker = HybridChunker(tokenizer=tokenizer)

def createChunks(doc):
    return list(chunker.chunk(dl_doc=doc))