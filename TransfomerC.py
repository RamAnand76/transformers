import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__().__init__()
        assert emb_size % num_heads == 0, "Embedding size must be divisible by the number of heads."

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads


        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.fc_out = nn.Linear(emb_size, emb_size)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x):
        batch_size, seq_length, emb_size = x.shape

        # Linear projections 
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Split into multiople heads
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Scaled dot-product attenntion
        energy = torch.einsum("bqhd, bkhd->bhqk", [queries, keys]) / self.scale
        attention = torch.softmax(energy, dim=-1)

        # Aggregate values
        out = torch.einsum("bhqk,bkhd->bqhd", [attention, values])

        # Concatenate heads
        out = out.reshape(batch_size, seq_length, self.emb_size)

        # Final linear layer
        out = self.fc_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, emb_size, ff_hidden_size, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(emb_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, ff_hidden_size, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(emb_size, num_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.ff = FeedForward(emb_size, ff_hidden_size, dropout)
        self.norm2 = nn.LayerNorm(emb_size)


    def forward(self, x):
        # Self-attention
        attention = self.attention(x)
        x = self.norm1(attention + x)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(ff_out + x)

        return x
    


class Transformer(nn.Module):
    def __init__(self, emb_size, num_heads, num_layers, ff_hidden_size, vocab_size, max_length, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, emb_size))
        self.layers = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(x.device)

        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        for layer in self.layers:
            x = layer(x)

        logits = self.fc_out(x)
        return logits
    





