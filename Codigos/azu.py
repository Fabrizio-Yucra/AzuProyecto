from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="gemma3:1b")

template = """
Eres un experto resolviendo preguntas frecuentes en una universidad de alto prestigio
Quiero que seas lo mas corto y claro posible
Aqui se encuentra la información necesaria: {informacion}

Aqui esta la pregunta para responder: {pregunta}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
while True:
	
	pregunta = input("Realiza una pregunta: ")
	if pregunta == "q":
		break
	informacion = retriever.invoke(pregunta)
	respuesta = chain.invoke({"informacion": informacion, "pregunta": pregunta})
	print(respuesta)

