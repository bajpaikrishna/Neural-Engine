from flask import Flask, request, jsonify
import torch
from src.models.efficient_transformer import EfficientTransformer
from src.inference.inference import make_prediction

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientTransformer(vocab_size=30522, embed_size=512, num_heads=8, hidden_size=2048, num_layers=6)
model = torch.quantization.convert(model)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = torch.tensor(data['input']).unsqueeze(0).to(device)
    predicted = make_prediction(model, input_data, device)
    return jsonify({'prediction': predicted.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
