from transformers import AutoTokenizer, AutoModel
import torch

# Load the tokenizer and the base transformer model
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

# Function to get text embeddings
def get_text_embeddings(texts):
    """
    Generate embeddings for a list of input texts.
    Args:
        texts (list of str): Input text strings.
    Returns:
        torch.Tensor: Embeddings of shape (num_texts, embedding_dim).
    """
    # Tokenize the inputs
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=65, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the last hidden state or pooled output as embeddings
        # Last hidden state: (batch_size, seq_length, hidden_dim)
        # Pooled output: (batch_size, hidden_dim)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
    return embeddings

# Example usage
texts = ["I feel so happy today!", "This is a sad moment."]
embeddings = get_text_embeddings(texts)
print("Embeddings shape:", embeddings.shape)
print(embeddings)
