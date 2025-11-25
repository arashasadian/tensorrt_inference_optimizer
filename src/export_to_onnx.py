from pathlib import Path

import torch
from torchvision.models import resnet18, ResNet18_Weights


def main():
    device = torch.device("cpu")

    # Load pretrained ResNet18
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device).eval()

    # Use a larger dummy batch so exporter "sees" batch > 1
    dummy = torch.randn(8, 3, 224, 224, device=device)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    out_path = models_dir / "resnet18_dynamic.onnx"

    print(f"Exporting ONNX model to {out_path}...")

    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=13,          # a bit newer opset, usually safer
        do_constant_folding=False, # avoid folding a fixed {1,512} reshape
        export_params=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
