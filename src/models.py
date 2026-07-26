import torch
import torch.nn as nn
import torchvision.models as models

class Stage1BinaryModel(nn.Module):
    """
    Stage 1 Classifier: Healthy (0) vs. DR Present (1).
    Uses Transfer Learning on ResNet18 or MobileNetV2.
    """
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super(Stage1BinaryModel, self).__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, 2) # 2 classes: Healthy, Diseased
            )
        elif backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v2(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.model(x)

class Stage2SeverityModel(nn.Module):
    """
    Stage 2 Classifier: Multi-class DR Severity (0: Mild, 1: Moderate, 2: Severe, 3: Proliferative).
    Uses Transfer Learning on EfficientNetB0 or ResNet50.
    """
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True):
        super(Stage2SeverityModel, self).__init__()
        self.backbone_name = backbone

        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 4) # 4 classes: Mild, Moderate, Severe, Proliferative
            )
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 4)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.model(x)

def build_stage1_model(backbone: str = "resnet18", pretrained: bool = True):
    return Stage1BinaryModel(backbone=backbone, pretrained=pretrained)

def build_stage2_model(backbone: str = "efficientnet_b0", pretrained: bool = True):
    return Stage2SeverityModel(backbone=backbone, pretrained=pretrained)
