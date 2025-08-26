import os
import torch
import torch.nn as nn
import json

class SimpleTransformer(nn.Module):
    """A simplified transformer model for testing"""
    def __init__(self, vocab_size=100, hidden_size=128, num_layers=2, num_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Create transformer layers
        self.attention_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.feed_forward_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        
        # Create output layer
        self.output_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for i in range(self.num_layers):
            # Simple forward pass
            x = x + self.attention_layers[i](x)
            x = x + self.feed_forward_layers[i](x)
            
        return self.output_head(x)

# Create directories
os.makedirs("checkpoints", exist_ok=True)

# Load vocabulary to get vocab size
with open("vocab.json", "r") as f:
    vocab = json.load(f)
    vocab_size = len(vocab)

# Create model with vocabulary size
model = SimpleTransformer(vocab_size=vocab_size)

# Save model checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size': vocab_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'num_heads': model.num_heads
    }
}, "checkpoints/base_model.pt")

print(f"Created base model checkpoint with vocab size: {vocab_size}")
