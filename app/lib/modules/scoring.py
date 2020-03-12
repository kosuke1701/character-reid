import torch

def metric_l2(tensor1, tensor2):
    return - torch.norm(tensor1 - tensor2, dim=1)

class AbstractScoringModel(object):
    def compute_score(self, feature_set1, feature_set2, mode):
        raise NotImplementedError()

class VectorEmbeddingScoringModel(AbstractScoringModel):
    def __init__(self, sim_func):
        self.sim_func = sim_func
    
    def compute_score(self, tensor1, tensor2, mode):
        if mode == "pair":
            d = tensor1.size(1)
            n1 = tensor1.size(0)
            n2 = tensor2.size(0)

            return self.sim_func(
                tensor1.repeat(1, n2).view(n1*n2, d),
                tensor2.repeat(n1, 1)
            ).view(n1, n2)
        elif mode == "batch":
            return self.sim_func(tensor1, tensor2)
        

