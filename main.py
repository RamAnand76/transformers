import numpy as np
import torch
import torch.nn as nn
import time
from TransfomerC import Transformer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the dataset
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Create a set of unique tokens (vocabulary)
    tokens = list(set(text))
    vocab_size = len(tokens)
    
    # Create mapping from token to index and index to token
    token_to_index = {token: idx for idx, token in enumerate(tokens)}
    index_to_token = {idx: token for idx, token in enumerate(tokens)}
    
    # Convert the text to indices
    data_indices = np.array([token_to_index[token] for token in text])

    return data_indices, vocab_size, token_to_index, index_to_token

# Parameters
file_path = 'myset.txt'
seq_length = 10  # Length of the sequences
num_samples = 1000  # Number of sequences to use for training

data_indices, vocab_size, token_to_index, index_to_token = load_data(file_path)

# Prepare training sequences
X_train = []
y_train = []

for i in range(len(data_indices) - seq_length):
    X_train.append(data_indices[i:i + seq_length])
    y_train.append(data_indices[i + seq_length])  # Next token

X_train = torch.tensor(X_train, dtype=torch.long).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

# Hyperparameters
emb_size = 128
num_heads = 8
num_layers = 4
ff_hidden_size = 512
max_length = seq_length
dropout = 0.1
epochs = 10
learning_rate = 3e-4

# Initialize model, loss, and optimizer
model = Transformer(emb_size, num_heads, num_layers, ff_hidden_size, vocab_size, max_length, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    start_time = time.time()
    
    for i in range(len(X_train)):
        optimizer.zero_grad()
        input_seq = X_train[i].unsqueeze(0)  # Add batch dimension
        target_seq = y_train[i].unsqueeze(0)  # Target is the next token
        
        output = model(input_seq)  # Forward pass
        
        # We only need to consider the last output for prediction
        output = output[:, -1, :]  # Shape: (batch_size, vocab_size)
        
        loss = criterion(output, target_seq)  # Compute loss
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        epoch_loss += loss.item()
        
        # Display ETA every 100 batches
        if (i + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            batches_left = len(X_train) - (i + 1)
            avg_time_per_batch = elapsed_time / (i + 1)
            estimated_time_left = avg_time_per_batch * batches_left
            
            print(f'Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(X_train)}, '
                  f'Loss: {epoch_loss / (i + 1):.4f}, '
                  f'ETA: {estimated_time_left / 60:.2f} minutes')

    print(f'Epoch {epoch + 1}/{epochs} completed, Average Loss: {epoch_loss / len(X_train):.4f}')

# Optionally save the model
#model.save_pretrained("./transfo")
