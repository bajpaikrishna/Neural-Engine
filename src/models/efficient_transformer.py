import torch
import torch.nn as nn
from linformer import Linformer

class EfficientTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers, seq_len=512, k=256):
        super(EfficientTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = Linformer(
            dim=embed_size, 
            seq_len=seq_len, 
            depth=num_layers, 
            heads=num_heads, 
            k=k
        )
        self.fc = nn.Linear(embed_size, vocab_size)

        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        embedded = self.embedding(x)
        output = self.transformer(embedded)
        output = self.fc(output)
        return self.dequant(output)

    def prune_model(self, amount=0.3):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=amount)
