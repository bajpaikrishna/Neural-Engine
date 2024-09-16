import torch
from src.models.student_model import StudentModel
from src.models.efficient_transformer import EfficientTransformer

def make_prediction(model, input_data, device):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        input_data = input_data.to(device)
        output = model(input_data)
    
    return output.argmax(dim=-1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the best model parameters from NAS
    model = EfficientTransformer(
        vocab_size=30522,
        embed_size=512,
        num_heads=8,  # Replace with best NAS parameters
        hidden_size=2048,
        num_layers=6  # Replace with best NAS parameters
    )
    model = torch.quantization.convert(model)

    input_data = torch.randint(0, 30522, (1, 512))
    prediction = make_prediction(model, input_data, device)
    print(f"Predicted label: {prediction}")
