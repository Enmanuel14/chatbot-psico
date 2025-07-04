from flask import Flask, request, jsonify
import g4f
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline("sentiment-analysis")

temas_psicologicos = [
    "ansiedad", "ansioso", "autoestima", "estrés", "estresado", "depresión", "deprimido",
    "emoción", "emociones", "triste", "feliz", "motivación", "motivado", "frustración", "frustrado",
    "trauma", "culpa", "culpable", "psicólogo", "psicología", "insomnio", "miedo", "soledad",
    "terapia", "rabia", "ira", "ayuda", "afrontar", "manejar", "solución", "superar", "me siento",
    "qué puedo hacer", "cómo me afecta", "cómo afrontarlo", "me afecta", "me está pasando"
]

@app.route('/', methods=['POST'])
def chatbot():
    data = request.get_json()
    pregunta = data.get('mensaje', '')
    
    resultado = classifier(pregunta)[0]
    label = resultado['label']
    score = resultado['score']
    contiene_tema = any(palabra in pregunta.lower() for palabra in temas_psicologicos)

    if (label == "NEGATIVE" and score > 0.7) or contiene_tema:
        respuesta = g4f.ChatCompletion.create(
            model=g4f.models.default,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente psicológico empático. Escuchas activamente, das apoyo emocional y consejos útiles, sin emitir diagnósticos clínicos."
                },
                {"role": "user", "content": pregunta}
            ]
        )
        return jsonify({"respuesta": respuesta})
    else:
        return jsonify({"respuesta": "⚠️ Puedo ayudarte si hablas sobre tus emociones, ansiedad, estrés u otros temas de salud mental."})

@app.route('/', methods=['GET'])
def index():
    return "Chatbot psicológico funcionando."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
