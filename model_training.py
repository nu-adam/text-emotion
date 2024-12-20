from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk, load_metric

# Load preprocessed datasets
train_data = load_from_disk("train_data")
test_data = load_from_disk("test_data")

# Load accuracy metric
accuracy_metric = load_metric("accuracy")

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=6)
model.config.max_position_embeddings = 65

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the model and tokenizer
model.save_pretrained("emotion_model")
