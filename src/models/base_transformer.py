import torch
import torch.nn as nn

class BaseTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers):
        super(BaseTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x, src_mask=None):
        embedded = self.embedding(x)
        output = self.transformer(embedded, src_mask)
        return self.fc(output)
