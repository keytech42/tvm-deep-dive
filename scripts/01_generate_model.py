import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple Convolutional layer to represent feature extraction
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # A fully connected layer to simulate classification (e.g. 10 classes)
        self.fc = nn.Linear(16 * 224 * 224, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def main():
    print("Initializing dummy PyTorch model...")
    model = DummyModel()
    model.eval()
    
    # Dummy input: batch_size=1, channels=3, height=224, width=224 (like ResNet)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    onnx_path = "dummy_model.onnx"
    print(f"Exporting model to ONNX format: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print("Export successful! The model is ready for TVMC compilation.")

if __name__ == "__main__":
    main()
