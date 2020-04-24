import configparser
import glob
import os
import _pickle as pic

import numpy as np
import torch

from ..common import AgglomerativeCluster, id_enroll

class AbstractMainModule(object):
    def __init__(self, max_batch_size):
        self.bs = max_batch_size

        config_fn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "config.ini"
        )
        self.config = configparser.ConfigParser()
        self.config.read(config_fn)

    def register_modules(self, image_loader, encoder_model, scoring_model):
        self.loader = image_loader
        self.encoder = encoder_model
        self.scoring = scoring_model
    
    def _load_thumbnail(self, lst_filenames, size, callback=None):
        self.loader.load_thumbnail(lst_filenames, size, callback)
    
    def _get_embedding(self, filenames, callback=None):
        assert len(filenames) > 0, "No filename is given to _get_embedding()"
        embs = []
        for i_start in range(0, len(filenames), self.bs):
            i_end = i_start + self.bs
            batch_fn = filenames[i_start:i_end]
            batch_img = self.loader.load_images(batch_fn)
            emb = self.encoder.encode(batch_img)
            embs.append(emb)
            if callback is not None:
                callback(i_start, len(filenames))
        if len(embs) > 1:
            embs = self.encoder.concatenate_emb(*embs)
        else:
            embs = embs[0]

        return embs
    
    def _get_list_of_all_image_files_directory(self, directory):
        extensions = self.config["common"]["image_file_extensions"].split(",")

        all_dir_files = glob.glob(os.path.join(directory, "**", "*")) + \
            glob.glob(os.path.join(directory, "*"))
        image_files = [fn for fn in all_dir_files 
            if os.path.isfile(fn) and fn.split(".")[-1].lower() in extensions]

        return image_files
    
    def get_similarity_result(self, filenames1, filenames2):
        with torch.no_grad():
            embs1 = self._get_embedding(filenames1)
            embs2 = self._get_embedding(filenames2)

            score_vec = self.scoring.compute_score(embs1, embs2, "batch")
            score_vec = np.array(score_vec.tolist())
        return score_vec
    
    def get_clustering_result(self, filenames):
        with torch.no_grad():
            # Compute distance matrix.
            embs = self._get_embedding(filenames)

            score_matrix = self.scoring.compute_score(embs, embs, "pair")
            distance_matrix = torch.max(score_matrix)[0] - score_matrix
            distance_matrix = np.array(distance_matrix.tolist())

        # Conduct clustering.
        model = AgglomerativeCluster()
        model.fit(distance_matrix)

        return model.get_range(), model.get_clusters
    
    def get_identification_result(self, enroll_filenames, target_filenames, mode, callback=None):
        """
        Arguments:
            enroll_filenames (list) -- List of lists of image filenames of each character.
            target_filenames (list) -- List of image filenames.
            mode (str) -- "Max" or "Avg"
        Return:
            id_scores - An numpy.ndarray of size (M x N) which contains identification scores between
                M target images and N known characters.
        """
        with torch.no_grad():
            if callback is not None:
                total_n_filenames = sum([len(_) for _ in enroll_filenames])
                last_i_fn = None
                last_n_fn = None
                n_processed = 0
                def _callback(i_fn, n_fn):
                    nonlocal n_processed, callback, last_i_fn, last_n_fn
                    if (last_i_fn is None) or (last_i_fn >= i_fn):
                        if last_n_fn is not None:
                            n_processed += last_n_fn - last_i_fn
                        last_i_fn = 0
                        last_n_fn = n_fn
                    n_processed += i_fn - last_i_fn
                    last_i_fn = i_fn

                    callback(0, n_processed, total_n_filenames)
            else:
                _callback = None

            # Embedding enroll images
            enroll_embs = []
            for i_fn, filenames in enumerate(enroll_filenames):
                enroll_embs.append(self._get_embedding(filenames, callback=_callback))
            if callback is not None:
                callback(0, len(enroll_filenames)-1, len(enroll_filenames))
            
            target_embs = self._get_embedding(target_filenames)
            if callback is not None:
                callback(1)

            scores = []
            for i_char, char_embs in enumerate(enroll_embs):
                scores.append(self.scoring.compute_score(char_embs, target_embs, "pair"))
                if callback is not None:
                    callback(2, i_char, len(enroll_embs))
            scores = id_enroll(scores, mode)
            callback(3)
        
        return scores

