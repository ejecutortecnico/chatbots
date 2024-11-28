from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import numpy as np
import pandas as pd

df = pd.read_csv("dialogos.csv")
train_data = df["question"]

train_labels = df["answer"]

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(train_labels)

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data)
train_sequences = tokenizer.texts_to_sequences(train_data)

train_sequences = keras.preprocessing.sequence.pad_sequences(train_sequences)
#Sequential groups a linear stack of layers into a tf.keras.Model.
model = keras.models.Sequential()

model.add(keras.layers.Embedding(len(tokenizer.word_index) + 1, 100, input_length=train_sequences.shape[1]))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(len(train_labels), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_sequences, encoded_labels, epochs=50)

def generate_response(text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=train_sequences.shape[1])
    prediction = model.predict(sequence)
    predicted_label = np.argmax(prediction)
    response = label_encoder.inverse_transform([predicted_label])[0]
    return response

while True:
    user_input = input("Enter a message: ")
    if user_input == "salir":
         print("Chatbot: ¡Adiós!")
         break
    response = generate_response(user_input)
    print("ChatBot: ", response)