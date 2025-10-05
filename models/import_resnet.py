# export_resnet.py
import argparse
import torch
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    resnet152,
    ResNet152_Weights,
    resnext50_32x4d,
    ResNeXt50_32X4D_Weights,
    resnext101_32x8d,
    ResNeXt101_32X8D_Weights,
    wide_resnet50_2,
    Wide_ResNet50_2_Weights,
    wide_resnet101_2,
    Wide_ResNet101_2_Weights,
)

VARIANTS = {
    "resnet18": (resnet18, ResNet18_Weights),
    "resnet34": (resnet34, ResNet34_Weights),
    "resnet50": (resnet50, ResNet50_Weights),
    "resnet101": (resnet101, ResNet101_Weights),
    "resnet152": (resnet152, ResNet152_Weights),
    "resnext50_32x4d": (resnext50_32x4d, ResNeXt50_32X4D_Weights),
    "resnext101_32x8d": (resnext101_32x8d, ResNeXt101_32X8D_Weights),
    "wide_resnet50_2": (wide_resnet50_2, Wide_ResNet50_2_Weights),
    "wide_resnet101_2": (wide_resnet101_2, Wide_ResNet101_2_Weights),
}


def main():
    parser = argparse.ArgumentParser(
        description="Export a ResNet to TorchScript (trace)."
    )
    parser.add_argument(
        "--model",
        choices=list(VARIANTS.keys()),
        default="resnet18",
        help="Which ResNet variant to use.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained ImageNet weights. Default: use DEFAULT weights.",
    )
    parser.add_argument(
        "--size", type=int, default=224, help="Square input size H=W. Default: 224."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for tracing. Default: 1."
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output .pt filename. Default: <model>.pt"
    )
    args = parser.parse_args()

    ctor, WeightsEnum = VARIANTS[args.model]
    weights = None if args.no_pretrained else WeightsEnum.DEFAULT

    model = ctor(weights=weights).eval()

    example_input = torch.randn(args.batch_size, 3, args.size, args.size)

    traced = torch.jit.trace(model, example_input)
    out_path = args.out or f"{args.model}.pt"
    traced.save(out_path)

    print(f"[OK] Traced model saved to: {out_path}")
    if weights is not None:
        print("[Tip] For inference, use the preprocessing tied to these weights:")
        print("      preprocess = weights.transforms()")
        print("      x = preprocess(pil_image)  # returns a normalized tensor")


if __name__ == "__main__":
    main()
