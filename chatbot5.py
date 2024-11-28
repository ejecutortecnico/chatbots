# Instalar las bibliotecas necesarias:
# pip install transformers torch

from transformers import pipeline

# Crear el pipeline de generación de texto usando DialoGPT
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")

def iniciar_chat():
    print("¡Hola! Soy un chatbot avanzado. Escribe 'adiós' para salir.")
    chat_historia = ""  # Inicializar historial de conversación

    while True:
        entrada = input("Tú: ")
        if entrada.lower() in ['adiós', 'chau', 'nos vemos']:
            print("ChatBot: ¡Adiós! Cuídate.")
            break

        # Concatenar la entrada del usuario al historial de la conversación
        chat_historia += f"Usuario: {entrada}\nChatBot:"
        respuesta = chatbot(chat_historia, max_length=100, pad_token_id=50256)

        # Obtener y mostrar la respuesta generada
        respuesta_texto = respuesta[0]["generated_text"][len(chat_historia):]
        print("ChatBot:", respuesta_texto.strip())

        # Actualizar el historial de conversación con la respuesta del bot
        chat_historia += respuesta_texto + "\n"

# Ejecutar el chatbot
iniciar_chat()
