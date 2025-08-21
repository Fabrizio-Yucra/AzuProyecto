from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from lector_pdf import retriever  

model = OllamaLLM(model="gemma3:1b")


template = """
Eres AZU, un asistente ROBOT experto en responder preguntas sobre la Universidad Católica Boliviana San Pablo,
tienes una personalidad extrovertida, enérgica y te gusta tu trabajo (Solo responde quien eres si te lo piden).
Usa únicamente el contexto proporcionado. 
- Responde SOLO con la información que esté claramente en la misma sección del contexto. 
- Si la pregunta pide (carreras, requisitos, becas, descuentos), devuelve una lista con viñetas.
- Si el contexto incluye más de una sección, usa SOLAMENTE la parte que responda directamente a la pregunta. Ignora secciones no relacionadas aunque aparezcan en el contexto.
- No inventes ni mezcles secciones.
- No adjuntes links directamente di que pueden preguntar a marketing para mas informacion.
- Si no encuentras la respuesta, di: "No está muy clara la pregunta."

Contexto:
{informacion}

Historial de conversación:
{chat_history}

Pregunta:
{pregunta}

Respuesta clara, precisa y concisa:
"""

prompt = ChatPromptTemplate.from_template(template)

# Memoria de la conversación
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="pregunta",
    output_key="respuesta",
    return_messages=True
)

# Cadena con memoria
chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    output_key="respuesta"
)

print("Asistente Universitario Iniciado (Escribe 'q' para salir)\n")

while True:
    pregunta = input(" Pregunta: ")
    if pregunta.lower() == "q":
        print(" Saliendo del asistente...")
        break
    
    informacion = retriever(pregunta)
    respuesta = chain.invoke({"informacion": informacion, "pregunta": pregunta})
    print("\n💡 Respuesta:\n", respuesta["respuesta"], "\n")
