import argparse
import torch
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
    vit_l_16,
    ViT_L_16_Weights,
)

VARIANTS = {
    "vit_b_16": (vit_b_16, ViT_B_16_Weights),
    "vit_l_16": (vit_l_16, ViT_L_16_Weights),
}


def main():
    parser = argparse.ArgumentParser(
        description="Export a Vision Transformer to TorchScript (script or trace)."
    )
    parser.add_argument(
        "--model",
        choices=list(VARIANTS.keys()),
        default="vit_b_16",
        help="Which ViT variant to use.",
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
    parser.add_argument(
        "--mode",
        choices=["script", "trace"],
        default="script",
        help="TorchScript export mode. Default: script (recommended).",
    )
    args = parser.parse_args()

    ctor, weights_enum = VARIANTS[args.model]
    weights = None if args.no_pretrained else weights_enum.DEFAULT

    model = ctor(weights=weights).eval()

    if args.mode == "script":
        scripted = torch.jit.script(model)
        export = scripted
    else:
        example_input = torch.randn(args.batch_size, 3, args.size, args.size)
        # check_trace=False avoids false positives on graph equality with ViT.
        traced = torch.jit.trace(model, example_input, check_trace=False)
        export = traced
    out_path = args.out or f"{args.model}.pt"
    export.save(out_path)

    print(f"[OK] TorchScript ({args.mode}) model saved to: {out_path}")
    if weights is not None:
        print("[Tip] For inference, use the preprocessing tied to these weights:")
        print("      preprocess = weights.transforms()")
        print("      x = preprocess(pil_image)  # returns a normalized tensor")


if __name__ == "__main__":
    main()
