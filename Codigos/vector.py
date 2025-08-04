from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Leer el CSV desde la ruta absoluta
df = pd.read_csv(
    r"C:\Users\FAYM\Documents\Semestre 2-2025\AzuProyecto\Codigos\inscripciones_2010.csv"
)

# Configurar embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

# Carpeta donde se guardará la base
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Crear lista de documentos
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["nombre_carrera"] + " " + row["tipo_inscripcion"],
            metadata={
                "fecha_inscripcion": row["fecha_inscripcion"],
                "anio_inscripcion": row["anio_inscripcion"],
                "sexo": row["sexo"],
                "pais_nacionalidad": row["pais_nacionalidad"],
                "correlativo_estudiante": row["correlativo_estudiante"],
            },
        )
        ids.append(str(i))
        documents.append(document)

# Inicializar Chroma
vector_store = Chroma(
    collection_name="Numero_de_inscritos",
    persist_directory=db_location,
    embedding_function=embeddings,
)

# 🔹 Dividir en lotes para evitar el error de batch size
if add_documents:
    batch_size = 5000  # debe ser menor que 5461
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)

# Crear retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
