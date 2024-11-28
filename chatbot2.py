import csv

respuestas = {}

with open("datoschat.csv", "r") as archivo:
    lector = csv.DictReader(archivo)
    for filas in lector:
        respuestas[filas["pregunta"]] = filas["respuesta"]

# Función para obtener la respuesta
def obtener_respuesta(mensaje):
    mensaje = mensaje.lower()
    if mensaje in respuestas:
        return respuestas[mensaje]
    else:
        with open("datoschat.csv", "a", newline="") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow([mensaje, "Pregunta por responder"])  # Escribir encabezados
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