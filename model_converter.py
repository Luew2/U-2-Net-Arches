import os
import torch
import torch.onnx
from model import U2NET

def find_latest_model(folder):
    print(f"Listing files in {folder}:")
    for file in os.listdir(folder):
        print(f"- {file}")

    models = [file for file in os.listdir(folder) if file.endswith('.pth')]
    if not models:
        raise ValueError(f"No models found in {folder}")
    latest_model = max(models, key=lambda x: os.path.getctime(os.path.join(folder, x)))
    return os.path.join(folder, latest_model)

script_folder = os.path.dirname(os.path.realpath(__file__))
model_folder = os.path.join(script_folder, "background_remover_model/u2net")
model_path = find_latest_model(model_folder)
onnx_model_path = "u2net.onnx"
batch_size = 1

torch_model = U2NET(3, 1)
torch_model.load_state_dict(torch.load(model_path))
torch_model.eval()

x = torch.randn(batch_size, 3, 320, 320, requires_grad=True)
torch_out = torch_model(x)

torch.onnx.export(
    torch_model,
    x,
    onnx_model_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"Model exported to {onnx_model_path}")
