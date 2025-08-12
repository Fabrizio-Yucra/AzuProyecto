import pyaudio
import audioop
from vosk import Model, KaldiRecognizer
import json
import time

def transcribir_por_energia(model_path="model/vosk", sample_rate=16000, threshold=500):
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, sample_rate)

    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16,
                      channels=1,
                      rate=sample_rate,
                      input=True,
                      frames_per_buffer=1024)
    stream.start_stream()

    print("🟢 Esperando voz...")

    silencio_contador = 0
    voz_activa = False
    buffer_audio = []

    try:
        while True:
            frame = stream.read(1024, exception_on_overflow=False)
            rms = audioop.rms(frame, 2)  # 2 bytes por muestra (paInt16)

            if rms > threshold:
                if not voz_activa:
                    print("🎙️ Voz detectada...")
                    voz_activa = True
                    buffer_audio = []
                buffer_audio.append(frame)
                silencio_contador = 0
            else:
                if voz_activa:
                    silencio_contador += 1
                    buffer_audio.append(frame)
                    if silencio_contador > 15:  # ~0.5 segundos de silencio
                        voz_activa = False
                        audio_data = b''.join(buffer_audio)
                        if recognizer.AcceptWaveform(audio_data):
                            result = json.loads(recognizer.Result())
                            print("📝 Transcripción:", result.get("text", ""))
                        else:
                            partial = json.loads(recognizer.PartialResult())
                            print("📝 Parcial:", partial.get("partial", ""))
                        print("🔴 Silencio detectado. Esperando voz...")
                        buffer_audio = []
                        silencio_contador = 0
    except KeyboardInterrupt:
        print("⏹️ Finalizado por el usuario.")
    finally:
        stream.stop_stream()
        stream.close()
        mic.terminate()