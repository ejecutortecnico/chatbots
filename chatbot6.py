import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 1. Cargar datos
def load_data(file_path):
    """Carga el dataset desde un archivo JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

data = load_data("dataset.json")
print(f"Datos cargados: {len(data)} ejemplos")

# Convertir datos a formato compatible con Hugging Face
dialogues = [{"input_text": d["input_text"], "target_text": d["target_text"]} for d in data]
dataset = Dataset.from_list(dialogues)

# Dividir en entrenamiento y evaluación (80% - 20%)
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Verificar estructura del dataset
print("Estructura de train_dataset:")
print(train_dataset)
print("Ejemplo de datos:")
print(train_dataset[0])

# Asegurarse de que las columnas tienen los nombres correctos
if "input_text" not in train_dataset.column_names or "target_text" not in train_dataset.column_names:
    train_dataset = train_dataset.rename_columns({
        "input": "input_text",
        "response": "target_text"
    })
    eval_dataset = eval_dataset.rename_columns({
        "input": "input_text",
        "response": "target_text"
    })

# 2. Cargar el modelo y el tokenizador
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Preprocesar datos
def preprocess_function(examples):
    """Tokeniza los datos de entrada y salida."""
    inputs = [f"User: {inp}\nBot:" for inp in examples["input_text"]]
    targets = [resp for resp in examples["target_text"]]
    
    # Tokenización de entrada
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # Tokenización de salida (etiquetas)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Probar la función de preprocesamiento
sample_batch = train_dataset[:5]
print("Ejemplo de batch:", sample_batch)
preprocessed = preprocess_function(sample_batch)
print("Resultado del preprocesamiento:", preprocessed)

# Tokenización de datasets
train_tokenized = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['input_text', 'target_text']
)

eval_tokenized = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['input_text', 'target_text']
)

print(f"Ejemplos tokenizados: {len(train_tokenized)} entrenamiento, {len(eval_tokenized)} evaluación")

# 4. Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir="./chatbot_model",
    evaluation_strategy="epoch",  # Evaluar después de cada época
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    push_to_hub=False,
    fp16=True,  # Usar precisión mixta si hay una GPU compatible
    report_to="none"  # Desactiva el reporte a servicios como W&B por defecto
)

# 5. Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    tokenizer=tokenizer  # Importante para modelos generativos
)

# 6. Entrenar el modelo
print("Iniciando el entrenamiento...")
trainer.train()

# 7. Guardar el modelo entrenado
model.save_pretrained("./trained_chatbot")
tokenizer.save_pretrained("./trained_chatbot")

print("¡Entrenamiento completado y modelo guardado!")
