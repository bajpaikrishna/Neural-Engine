import torch
from src.models.efficient_model import EfficientTransformer
from src.training.train import train_model
from src.utils.data_loader import get_data
from src.utils.config import get_config

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config = get_config()
    
    # Initialize model
    model = EfficientTransformer(vocab_size=30522, embed_size=512, num_heads=8, hidden_size=2048, num_layers=6)
    
    # Get data
    data_loader = get_data(config['batch_size'])
    
    # Train the model
    train_model(model, data_loader, config['num_epochs'], device)
