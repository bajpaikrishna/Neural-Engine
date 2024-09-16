import torch
from src.training.train import train_model
from src.models.base_model import BaseTransformer
from src.utils.data_loader import get_data

def test_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab_size = 30522
    embed_size = 512
    num_heads = 8
    hidden_size = 2048
    num_layers = 6
    num_epochs = 2  # Reduce epochs for testing
    
    model = BaseTransformer(vocab_size, embed_size, num_heads, hidden_size, num_layers)
    data_loader = get_data(batch_size=8)  # Small batch for testing
    
    train_model(model, data_loader, num_epochs, device)
    print("Training Test Passed")

import logging

# Set up logging to files
logging.basicConfig(
    filename='logs/train_logs.txt',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def train_model(model, data_loader, num_epochs, device):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log the loss after each epoch
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader)}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader)}")


if __name__ == "__main__":
    test_training()
