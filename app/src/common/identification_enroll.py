import torch

import numpy as np

def id_enroll(scores, mode):
    """
    Parameters:
        scores - A list of PyTorch FloatTensor. N-th tensor contains matching scores
            between images of an N-th character and target images
            (size: num_character_image x num_target_image).
        mode - Either "Max" or "Avg".
    Return:
        id_scores - An numpy.ndarray of size (M x N) which contains identification scores between
            M target images and N known characters.
    """
    if mode == "Max":
        scores = [torch.max(_, dim=0, keepdim=True)[0] for _ in scores]
    elif mode == "Avg":
        scores = [torch.mean(_, dim=0, keepdim=True) for _ in scores]
    else:
        raise Exception(f"Illegal model: {mode}")
    
    scores = torch.t(torch.cat(scores, dim=0)) # n_character x n_target

    return np.array(scores.tolist())

if __name__=="__main__":
    scores = [
        torch.FloatTensor([[0.1, 0.8], [0.05, 0.9]]),
        torch.FloatTensor([[0.9, 0.0], [0.8, 0.1], [0.7, 0.1]])
    ]
    print(id_enroll(scores, "Avg"))