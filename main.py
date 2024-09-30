import numpy as np
import torch
import torch.nn as nn
from TransfomerC import Transformer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate dummy data
vocab_size = 100
seq_length = 10
num_samples = 1000

X_train = np.random.randint(0, vocab_size, (num_samples, seq_length))
y_train = np.random.randint(0, vocab_size, (num_samples, seq_length))

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
    for i in range(len(X_train)):
        optimizer.zero_grad()
        input_seq = X_train[i].unsqueeze(0)  # Add batch dimension
        target_seq = y_train[i].unsqueeze(0)
        
        output = model(input_seq)  # Forward pass
        
        # Flatten the output and target tensors
        output = output.view(-1, vocab_size)
        target_seq = target_seq.view(-1)
        
        loss = criterion(output, target_seq)  # Compute loss
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X_train)}')

#model.save_pretrained("./transfo")

