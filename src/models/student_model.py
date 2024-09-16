import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers):
        super(StudentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer(embedded)
        return self.fc(output)

class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, target):
        loss = self.criterion(F.log_softmax(student_logits / self.temperature, dim=-1),
                              F.softmax(teacher_logits / self.temperature, dim=-1))
        return loss * (self.temperature ** 2) + F.cross_entropy(student_logits, target)
