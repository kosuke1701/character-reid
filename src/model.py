import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class EmbModel(nn.Module):
    def __init__(self, dim_emb, cls_num):
        """
        Args:
            dim_emb (int) : Dimension size of embeddings of images and classes.
            cls_num (int) : Number of classes.
        """
        super(EmbModel, self).__init__()

        self.ft_model = models.resnet18(pretrained=True)
        self.ft_model.fc = nn.Linear(512, dim_emb)

        self.cls_emb = nn.Parameter(torch.FloatTensor(cls_num, dim_emb))
        nn.init.xavier_normal_(self.cls_emb)
        self.normalize()

    def normalize(self):
        """Normalize class embeddings. Each embeddings' L2 norm will be 1."""
        with torch.no_grad():
            self.cls_emb.div_(torch.norm(self.cls_emb, dim=1, keepdim=True))

    def calc_data_embedding(self, images):
        u"""Calculate embeddings for input images.

        Args:
            images (4-dim FloatTensor) : Dimension is (Batch, Channel, H, W).
        Return:
            FloatTensor : Image embeddings. (Batch, Dim_embedding)
                Note that they are normalize to have unit L2 norm.
        """
        image_emb = self.ft_model(images)
        image_emb = image_emb / torch.norm(image_emb, dim=1, keepdim=True)
        return image_emb

    def forward(self, images):
        image_emb = self.calc_data_embedding(images)
        
        return image_emb