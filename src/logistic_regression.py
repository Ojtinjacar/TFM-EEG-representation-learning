import torch.nn as nn

class MLP_Classifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MLP_Classifier, self).__init__()
        self.classifier= nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            nn.ReLU(),
            nn.Linear(n_features // 2, n_classes)
        )

    def forward(self, x):
        return self.classifier(x)
