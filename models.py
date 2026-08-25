"""CNN model zoo: a compact custom baseline plus standard torchvision backbones,
all re-headed to the current dataset's number of classes / input channels."""

import torch
import torch.nn as nn
import torchvision.models as tvm


class SimpleCNN(nn.Module):
    """Compact CNN baseline — fast to train, useful as a control / surrogate model."""

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.out_channels = 128
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


MODEL_REGISTRY = {
    "simplecnn": "Custom lightweight CNN (fast baseline / surrogate)",
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "vgg16": "VGG-16",
    "mobilenet_v2": "MobileNetV2 (lightweight)",
    "efficientnet_b0": "EfficientNet-B0 (lightweight)",
}


def build_model(name, num_classes, in_channels=3, pretrained=False):
    name = name.lower()

    if name == "simplecnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)

    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "resnet34":
        m = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "vgg16":
        m = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            m.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        m.classifier[6] = nn.Linear(4096, num_classes)
        return m

    if name == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            m.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
        return m

    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            m.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    raise ValueError(f"Unknown model: {name}")


def get_feature_extractor(model, name):
    """Return (feature_module, feature_channels): the part of the network that
    outputs a spatial feature map, before global pooling / classifier. Used to
    splice in architectural enhancements (pyramid pooling, feature denoising)."""
    name = name.lower()
    if name == "simplecnn":
        return model.features, model.out_channels
    if name in ("resnet18", "resnet34"):
        modules = list(model.children())[:-2]  # drop avgpool + fc
        return nn.Sequential(*modules), model.fc.in_features
    if name == "vgg16":
        return model.features, 512
    if name == "mobilenet_v2":
        return model.features, model.last_channel
    if name == "efficientnet_b0":
        return model.features, model.classifier[1].in_features
    raise ValueError(f"Unknown model: {name}")
