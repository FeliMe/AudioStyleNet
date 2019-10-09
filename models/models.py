import torch.nn as nn

from torchvision import models as torch_models


# Pre-trained ResNet18
class PreTrainedResNet18(nn.Module):
    def __init__(self):
        super(PreTrainedResNet18, self).__init__()

        self.model = torch_models.resnet18(pretrained=True)

        for i, child in enumerate(self.model.children()):
            if i < len(list(self.model.children())) - 1:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)
