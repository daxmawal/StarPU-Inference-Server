import torchvision.models as models
import torch

model = models.resnet18(pretrained=True)
model.eval()
example_input = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example_input)
traced.save("resnet18.pt")
