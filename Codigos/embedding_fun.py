from langchain_ollama.embeddings import OllamaEmbeddings  # ✅ nuevo import actualizado

def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text:v1.5")
