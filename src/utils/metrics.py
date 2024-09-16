import torch

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=-1)
    correct = (preds == labels).float()
    acc = correct.sum() / len(correct)
    return acc
