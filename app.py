from flask import Flask, request, jsonify
from transformers import pipeline

# Cargar el modelo solo una vez (fuera de la función)
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

app = Flask(__name__)

# Filtros de palabras sensibles
palabras_clave = [
    "me quiero morir", "no quiero vivir", "suicidarme", "matarme", "morirme",
    "quitarme la vida", "no puedo más", "todo es una mierda", "odio mi vida",
    "quiero desaparecer", "me siento vacío", "me siento sola", "no valgo nada",
    "ya no tiene sentido", "quiero rendirme", "no tengo esperanza",
    "no le importo a nadie", "estoy desesperado", "no hay salida"
]

@app.route('/')
def home():
    return "Chatbot psicológico con filtros activos está funcionando."

@app.route('/analizar', methods=['POST'])
def analizar():
    data = request.get_json()
    texto = data.get("texto", "")

    if not texto:
        return jsonify({"error": "Texto no proporcionado"}), 400

    # Analizar sentimiento
    resultado = classifier(texto)

    # Buscar coincidencias exactas (filtros activados)
    coincidencias = [palabra for palabra in palabras_clave if palabra in texto.lower()]

    return jsonify({
        "sentimiento": resultado,
        "alertas_detectadas": coincidencias
    })

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=port)
