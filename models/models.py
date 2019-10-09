import torch.nn as nn

from torch.nn.modules import Sequential
from torchvision import models as torch_models


resnet18 = torch_models.resnet18(pretrained=True)

landmark_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(68 * 2, 8),
    nn.Softmax(dim=1)
)
