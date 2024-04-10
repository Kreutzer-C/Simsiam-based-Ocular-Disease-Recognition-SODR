import torch
import torch.nn as nn
from torchvision.models import inception_v3
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
    def __init__(self, out_dim, in_dim=2048, hidden_dim=512):  # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def get_backbone(backbone, castrate=True):
    if backbone == 'resnet18' or backbone == 'resnet50':
        backbone = eval(f"{backbone}(zero_init_residual=True)")
    elif backbone == 'icptv3':
        backbone = inception_v3(pretrained=False, aux_logits=False)
        print(backbone)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

class FTmodel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args.backbone)
        self.projector = projection_MLP(self.backbone.output_dim)
        self.encoder = nn.Sequential(  # f encoder
            self.backbone,
            self.projector
        )
        self.prediction = prediction_MLP(args.num_class)

        if args.forzen_en:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x1, x2):
        ft1 = self.encoder(x1)
        ft2 = self.encoder(x2)

        out1 = self.prediction(ft1)
        out2 = self.prediction(ft2)

        return out1, out2