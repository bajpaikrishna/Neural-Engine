import torch
from src.models.base_model import BaseTransformer

def test_base_transformer():
    vocab_size = 30522
    embed_size = 512
    num_heads = 8
    hidden_size = 2048
    num_layers = 6

    model = BaseTransformer(vocab_size, embed_size, num_heads, hidden_size, num_layers)
    inputs = torch.randint(0, vocab_size, (32, 512))
    
    outputs = model(inputs)
    assert outputs.shape == (32, 512, vocab_size), "Output shape mismatch"
    print("Base Transformer Test Passed")
