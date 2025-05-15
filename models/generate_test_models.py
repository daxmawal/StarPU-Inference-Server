import torch
import torch.nn as nn
import os

os.makedirs("test_models", exist_ok=True)

models_info = {
    "float32": (
        torch.float32,
        torch.randn(1, 10, dtype=torch.float32),
        nn.Linear(10, 5)
    ),
    "float16": (
        torch.float16,
        torch.randn(1, 10, dtype=torch.float16),
        nn.Linear(10, 5).half()
    ),
    "bfloat16": (
        torch.bfloat16,
        torch.randn(1, 10, dtype=torch.bfloat16),
        nn.Linear(10, 5).to(dtype=torch.bfloat16)
    ),
    "int64": (
        torch.int64,
        torch.tensor([[1, 2, 3]], dtype=torch.int64),
        nn.Embedding(100, 8)
    ),
    "int32": (
        torch.int32,
        torch.randint(0, 100, (1, 10), dtype=torch.int32),
        nn.Sequential(nn.Identity())
    ),
    "int16": (
        torch.int16,
        torch.randint(0, 100, (1, 10), dtype=torch.int16),
        nn.Sequential(nn.Identity())
    ),
    "int8": (
        torch.int8,
        torch.randint(-128, 127, (1, 10), dtype=torch.int8),
        nn.Sequential(nn.Identity())
    ),
    "uint8": (
        torch.uint8,
        torch.randint(0, 255, (1, 10), dtype=torch.uint8),
        nn.Sequential(nn.Identity())
    ),
    "bool": (
        torch.bool,
        torch.randint(0, 2, (1, 10), dtype=torch.bool),
        nn.Sequential(nn.Identity())
    ),
    "char_ascii": (
        torch.int64,
        torch.tensor([[ord('a'), ord('b'), ord('c')]], dtype=torch.uint8).to(torch.int64),
        nn.Embedding(256, 8)
    ),
}

for name, (dtype, example_input, model) in models_info.items():
    model.eval()
    try:
        traced = torch.jit.trace(model, example_input)
        path = f"test_models/model_{name}.pt"
        traced.save(path)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"Failed for {name}: {e}")
