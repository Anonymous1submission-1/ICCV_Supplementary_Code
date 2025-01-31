import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, GPT2Model, GPT2Tokenizer

# Check if CUDA is available for GPU acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Vision-Enhanced Large Language Model (VLLM) Architecture
class VisionEnhancedLLM(nn.Module):
    def __init__(self):
        super(VisionEnhancedLLM, self).__init__()
        
        # Load pre-trained Vision Transformer (ViT) as the vision encoder
        self.vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
        
        # Load pre-trained GPT-2 as the language model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.language_model = GPT2Model.from_pretrained("gpt2").to(device)
        
        # Define fusion layer to merge vision and language features
        self.fusion_layer = nn.Linear(768 * 2, 768).to(device)
        
        # Define final prediction layer
        self.output_layer = nn.Linear(768, 30522).to(device)  # Vocabulary size of GPT-2
    
    def forward(self, image_inputs, text_inputs):
        # Extract visual features
        vision_features = self.vision_encoder(image_inputs.to(device)).last_hidden_state[:, 0, :]
        
        # Tokenize and process text inputs
        text_tokens = self.tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        text_features = self.language_model(**text_tokens).last_hidden_state[:, 0, :]
        
        # Concatenate vision and language features
        combined_features = torch.cat((vision_features, text_features), dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # Generate output tokens
        output_tokens = self.output_layer(fused_features)
        return output_tokens

# Initialize model
model = VisionEnhancedLLM()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Example forward pass
def example_run():
    dummy_image = torch.randn(1, 3, 224, 224).to(device)  # Dummy image tensor
    dummy_text = "A cat sitting on a table."  # Example text
    
    output = model(dummy_image, dummy_text)
    print("Model Output Shape:", output.shape)

example_run()
