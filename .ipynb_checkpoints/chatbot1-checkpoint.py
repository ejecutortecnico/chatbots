respuestas = {
        "hola": "¡Hola! ¿En qué puedo ayudarte?",
        "buenos días": "¡Buenos días! Espero que tengas un excelente día.",
        "adiós": "Adiós, ¡espero verte pronto!",
        "bye": "Adiós, ¡cuídate mucho!",
        "tu nombre": "Soy un chatbot creado en Python. Aún no tengo nombre 😅",
        "clima": "No tengo acceso a información en tiempo real, pero podrías consultar el pronóstico en línea."
    }

# Función para obtener la respuesta
def obtener_respuesta(mensaje):
    mensaje = mensaje.lower()
    if mensaje in respuestas:
        return respuestas[mensaje]
    else:
        return "Lo siento, no entiendo esa pregunta."

# Bucle de conversación
print("¡Hola! Soy un chatbot. Escribe 'salir' para terminar la conversación.")
while True:
    usuario = input("Tú: ")
    if usuario.lower() == "salir":
        print("Chatbot: ¡Adiós!")
        break
    respuesta = obtener_respuesta(usuario)
    print("Chatbot:", respuesta)