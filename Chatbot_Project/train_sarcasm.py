import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------------------------------
# Step 1: Load and Preprocess Data
# -------------------------------

# Load dataset (ensure correct file path)
df = pd.read_json("Sarcasm_Headlines_Dataset_v2/Sarcasm_Headlines_Dataset_v2.json", lines=True)

# Keep only necessary columns
df = df[["headline", "is_sarcastic"]]
df = df.rename(columns={"is_sarcastic": "label"})  # Rename column

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Function to tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["headline"], truncation=True, padding="max_length", max_length=128)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns (adjust column names as needed)
columns_to_remove = ["headline"]
if "__index_level_0__" in tokenized_dataset.column_names:
    columns_to_remove.append("__index_level_0__")
tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

# Rename the label column to "labels" (required by Trainer)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Split the dataset into training and validation
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"].select(range(5000))  # Reduce training samples
val_dataset = split_dataset["test"].select(range(1000))  # Reduce validation samples

print("Tokenization complete. Training samples:", len(train_dataset), "Validation samples:", len(val_dataset))

# -------------------------------
# Step 2: Fine-Tune DistilBERT for Sarcasm Detection
# -------------------------------

# Load pre-trained DistilBERT model for sequence classification (with 2 labels: non-sarcastic=0, sarcastic=1)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define a function to compute metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,  # Reduce epochs to speed up training
    per_device_train_batch_size=16,  # Increase batch size
    per_device_eval_batch_size=16,  # Increase batch size
    warmup_steps=250,  # Reduce warmup steps
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,  # Log less frequently
    evaluation_strategy="epoch",
    device="cuda" if torch.cuda.is_available() else "cpu",  # Enable GPU if available
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Save the trained model and tokenizer for later use in your chatbot
model.save_pretrained("trained_sarcasm_model")
tokenizer.save_pretrained("trained_sarcasm_model")
print("Model saved successfully!")
