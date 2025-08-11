from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from lector_pdf import retriever  # Activa esta línea

model = OllamaLLM(model="gemma3:1b")
#"gemma3:1b"
template = """
Eres un experto resolviendo preguntas frecuentes en una universidad de alto prestigio
Quiero que tus respuestas sean lo mas corto, claro posible y en español  
Aquí se encuentra la información necesaria: {informacion}

Aquí está la pregunta para responder: {pregunta}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    pregunta = input("Realiza una pregunta ('q' para salir): ")
    if pregunta.lower() == "q":
        break
    informacion = retriever(pregunta)
    #print(" CONTEXTO EXTRAÍDO:\n", informacion)
    
    respuesta = chain.invoke({"informacion": informacion, "pregunta": pregunta})
    print("\n Respuesta:\n", respuesta)