import os
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Load dataset
train_data = pd.read_csv("train.csv")

# Filter data for sarcasm class
sarcasm_data = train_data[train_data["class"] == "sarcasm"]

# Define target labels
class_dict = {
    'sarcasm': 1
}

# Map class labels to integers
sarcasm_data["class"] = sarcasm_data["class"].map(class_dict)

# Load pre-trained tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Tokenize input texts
tokenized_inputs = tokenizer(sarcasm_data["tweets"].tolist(), padding=True, truncation=True, return_tensors="tf")

# Convert labels to TensorFlow tensors
labels = tf.convert_to_tensor(sarcasm_data["class"].tolist())

# Prepare TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized_inputs), labels)).shuffle(len(sarcasm_data)).batch(16)

# Define optimizer, loss, and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Fine-tune the model
model.fit(train_dataset, epochs=1)

# Save the fine-tuned model
model.save_pretrained('fine_tuned_sarcasm_model')

















