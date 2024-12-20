from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("emotion_model", ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained("emotion_model")

# Create a pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Run inference on a sample text
text = "I feel so happy today!"
result = classifier(text)
print(result)
