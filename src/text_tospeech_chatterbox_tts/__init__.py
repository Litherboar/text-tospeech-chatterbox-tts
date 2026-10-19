
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando modelo...")

# Usar GPU si CUDA está disponible; de lo contrario, CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device=device
)

texto = (
    "Hola, bienvenidos a Inteligencia Artificial Generativa. "
    "Vamos a crear audios, imágenes, texto y, si el PC aguanta, videos."
)

print("Generando audio...")

audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7,  # Intensidad de la expresión
    cfg_weight=0.4,    # Control de la guía de generación
    temperature=0.8,   # Variabilidad de la generación
    repetition_penalty=1.2,  # Corrección del nombre del parámetro
    min_p=0.05,        # Filtrado de opciones poco probables
    top_p=1.0          # Muestreo de tokens
)

ta.save(
    "audio.wav",
    audio.cpu(),
    modelo.sr
)

print("Audio guardado como audio.wav")
