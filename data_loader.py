from datasets import load_dataset
from transformers import AutoTokenizer

# Load the dataset
dataset = load_dataset("emotion")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Tokenize function
def preprocess_data(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=65)

# Split and preprocess dataset
train_data = dataset['train'].map(preprocess_data, batched=True)
test_data = dataset['test'].map(preprocess_data, batched=True)

print(dataset)
print(dataset['train'][0])
print(train_data[0])

# Save preprocessed datasets for training and testing
train_data.save_to_disk("train_data")
test_data.save_to_disk("test_data")
