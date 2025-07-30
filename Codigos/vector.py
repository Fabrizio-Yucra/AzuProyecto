from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("Inscritos.csv")
embeddings = OllamaEmbeddings(model = "nomic-embed-text:v1.5")
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
	documents = []
	ids = []
	
	for i, row in df.iterrows():
		document = Document(
			metadata={"carrera": row["Carrera"], "inscritos": row["Inscritos"]},
			id=str(i)
		)
		ids.append(str(i))
		documents.append(document)
vector_store = Chrome(
	collection_name = "Numero_Inscritos",
	persist_directory=db_location,
	embedding_function=embeddings
)

if add_documents:
	vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
	search_kwargs={"k": 5}
)
