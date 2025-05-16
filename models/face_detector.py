import torch.nn as nn
import torchvision.models as models

class FaceDetectionModel(nn.Module):
    def __init__(self):
        super(FaceDetectionModel, self).__init__()

        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.feature_extractor = backbone.features

        # Регрессия (bounding box)
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )

        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        bbox = self.regression_head(features)
        classification = self.classifier_head(features)

        return classification, bbox

