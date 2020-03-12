import torch

class AbstractModel(object):
    def encode(self, image_set):
        raise NotImplementedError()
    
    def concatenate_emb(self, *args):
        raise NotImplementedError()

class SingleImageEncoder(AbstractModel):
    def __init__(self, device):
        self.device = device

    def get_model(self):
        raise NotImplementedError()

    def encode(self, image_set):
        model = self.get_model()
        model.eval()

        emb = model(image_set.to(self.device))

        return emb
    
    def concatenate_emb(self, *tensors):
        return torch.cat(tensors, dim=0)