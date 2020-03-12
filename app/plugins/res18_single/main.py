import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from torchvision import transforms, models

from model import Identity, Normalize
from lib.modules import AbstractMainModule, FullImageLoader, SingleImageEncoder, \
    VectorEmbeddingScoringModel, metric_l2

class Res18SingleEncoder(SingleImageEncoder):
    def __init__(self, device):
        super().__init__(device)

        dropout = 0.0
        emb_dim = 500
        model_fn = "plugins/res18_single/res18_0222.mdl"

        self.trunk = models.resnet18(pretrained=True)
        trunk_output_size = self.trunk.fc.in_features
        self.trunk.fc = Identity()

        self.model = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0.0 else Identity(),
            nn.Linear(trunk_output_size, emb_dim),
            Normalize()
        )

        saved_models = torch.load(model_fn, map_location="cpu")
        self.trunk.load_state_dict(saved_models["trunk"])
        self.model.load_state_dict(saved_models["embedder"])

        self.comp_model = nn.Sequential(self.trunk, self.model)
    
    def get_model(self):
        return self.comp_model


class MainModule(AbstractMainModule):
    def __init__(self, device="cpu", max_batch_size=100):
        super().__init__(max_batch_size)

        # Image loader
        trans = [
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        trans = transforms.Compose(trans)

        loader = FullImageLoader(trans)

        # Encoder
        encoder = Res18SingleEncoder(device)

        # Scoring model
        scoring = VectorEmbeddingScoringModel(metric_l2)

        self.register_modules(loader, encoder, scoring)