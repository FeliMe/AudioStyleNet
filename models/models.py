import torch.nn as nn

from torchvision import models as torch_models


# Pre-trained ResNet18
class PreTrainedResNet18(nn.Module):
    def __init__(self):
        super(PreTrainedResNet18, self).__init__()

        model = torch_models.resnet18(pretrained=True)

        for i, child in enumerate(model.children()):
            if i < len(list(model.children())) - 1:
                for param in child.parameters():
                    param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 8)

        self.model = model

    def forward(self, x):
        return self.model(x)
