# Initialize model
model = Transformer(emb_size, num_heads, num_layers, ff_hidden_size, vocab_size, max_length, dropout)
model.load_state_dict(torch.load("transformer_final.pth"))
model.to(device)
model.eval()  # Set the model to evaluation mode