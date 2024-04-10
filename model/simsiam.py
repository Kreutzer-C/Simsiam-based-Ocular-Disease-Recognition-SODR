import torch
import torch.nn as nn
from model.backbones import *

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}(zero_init_residual=True)")
    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

class SimSiam(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.backbone = get_backbone(args.backbone)
        self.projector = projection_MLP(in_dim=self.backbone.output_dim)

        self.encoder = nn.Sequential(  # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()

    def forward(self, x1, x2):
        f = self.encoder
        h =  self.predictor

        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        return p1, p2, z1.detach(), z2.detach()