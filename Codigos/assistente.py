import os
import sounddevice as sd
import soundfile as sf
import json
import pyttsx3
from vosk import Model, KaldiRecognizer
from lector_pdf import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from embedding_fun import get_embedding_function

# 🧠 Validación semántica con log
def es_semanticamente_valida(pregunta: str, embedding_fn, umbral: float = 0.028) -> tuple[bool, float]:
    pregunta = pregunta.strip()
    if not pregunta:
        return False, 0.0
    emb = embedding_fn.embed_query(pregunta)
    densidad = sum(abs(x) for x in emb) / len(emb)
    return densidad > umbral, densidad

# 📚 Prompt y modelo
VOSK_MODEL_PATH = "model/vosk"
BIP_START = "audios/bip.wav"
BIP_END = "audios/bip2.wav"

model = OllamaLLM(model="gemma3:1b")

template = """
Eres un asistente experto resolviendo preguntas frecuentes en una universidad de alto prestigio.
Quiero que tus respuestas sean cortas de no mas de 3 lineas, claras y en español.
Aquí se encuentra la información necesaria: {informacion}
Aquí está la pregunta para responder: {pregunta}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def play_sound(path):
    data, fs = sf.read(path)
    sd.play(data, fs)
    sd.wait()

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

def record_and_transcribe():
    play_sound(BIP_START)
    model_vosk = Model(VOSK_MODEL_PATH)
    recognizer = KaldiRecognizer(model_vosk, 16000)

    with sd.RawInputStream(samplerate=16000, channels=1, dtype='int16') as stream:
        print("🎙️ Grabando... habla ahora")
        for _ in range(80):  # ~5 segundos
            data, _ = stream.read(4096)
            data_bytes = bytes(data)
            if recognizer.AcceptWaveform(data_bytes):
                result = recognizer.Result()
                text = json.loads(result)["text"]
                play_sound(BIP_END)
                return text.strip()
        play_sound(BIP_END)
        return ""  # No se detectó voz útil

# 🚀 Ciclo principal con log de densidad
if __name__ == "__main__":
    embedding_fn = get_embedding_function()
    log_rechazadas = []

    while True:
        pregunta = record_and_transcribe()
        if not pregunta:
            print(" No se detectó voz útil. Intenta de nuevo.")
            continue
        if pregunta.lower() in ["salir", "q"]:
            break

        es_valida, densidad = es_semanticamente_valida(pregunta, embedding_fn)
        print(f"🔍 Densidad semántica: {densidad:.4f}")

        if not es_valida:
            print("⚠️ Pregunta inválida o sin sentido. Intenta con algo más específico.")
            log_rechazadas.append((pregunta, densidad))
            continue

        print(f"\n Pregunta: {pregunta}")
        informacion = retriever(pregunta)
        respuesta = chain.invoke({"informacion": informacion, "pregunta": pregunta})
        print("\n Respuesta:\n", respuesta)
        speak(respuesta)

    # 📝 Exportar preguntas rechazadas al final
    if log_rechazadas:
        with open("preguntas_rechazadas.log", "w", encoding="utf-8") as f:
            for texto, densidad in log_rechazadas:
                f.write(f"{densidad:.4f} | {texto}\n")
        print(f"\n📁 Se guardaron {len(log_rechazadas)} preguntas rechazadas en 'preguntas_rechazadas.log'")