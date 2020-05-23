import configparser
import json
import os
import sys
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-root", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    # Load EfficientDet
    ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
    config = configparser.ConfigParser()
    config.read(os.path.join(ROOT_DIR, "config.ini"))
    print(list(config.keys()))
    effdet_dir = config["DEFAULT"]["EfficientDetDirectory"]
    effdet_weight = config["DEFAULT"]["EfficientDetWeight"]
    effdet_coeff = int(config["DEFAULT"]["EfficientDetCoefficient"])
    effdet_input_size = int(config["DEFAULT"]["EfficientDetInputSize"])
    effdet_thr = float(config["DEFAULT"]["EfficientDetThreshold"])
    effdet_iou_thr = float(config["DEFAULT"]["EfficientDetIoUThreshold"])

    sys.path.append(effdet_dir)
    from backbone import EfficientDetBackbone
    from efficientdet.utils import BBoxTransform, ClipBoxes
    from utils.utils import preprocess, invert_affine, postprocess
    
    model = EfficientDetBackbone(
        compound_coef = effdet_coeff,
        num_classes = 1,
        ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        scales = [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]
    )
    if args.cuda:
        model.load_state_dict(torch.load(effdet_weight))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(effdet_weight, \
            map_location=torch.device("cpu")))
    model.eval()

    def get_face_position(fn):
        _, fimg, meta = preprocess(fn, max_size=effdet_input_size)
        x = torch.from_numpy(fimg[0]).float().unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        if args.cuda:
            x = x.cuda()

        with torch.no_grad():
            _, reg, clss, anchors = model(x)

            rbox = BBoxTransform()
            cbox = ClipBoxes()

            out = postprocess(x, anchors, reg, clss, rbox, cbox, \
                effdet_thr, effdet_iou_thr)
            out = invert_affine(meta, out)
        
        lst_face_bbox = []
        for i_detect in range(len(out[0]["rois"])):
            lst_face_bbox.append(
                [int(val) for val in out[0]["rois"][i_detect]]
            )
        return lst_face_bbox

    # Preprocess image
    sys.path.append(os.path.join(ROOT_DIR, "src"))
    from util import _create_index_image_map

    idx_img_map = _create_index_image_map(args.image_root)

    data = json.load(open(args.dataset))

    set_fn = set([])
    face_position_data = []
    for i_split, split in enumerate(data):
        print(f"Processing {i_split}-th data split.")

        image_ids = set([i_img for i_char, img_lst in split for i_img in img_lst])
        n_images = len(image_ids)
        positions = {}
        bar = tqdm(total=n_images)
        for i_img in image_ids:
            fn = idx_img_map[i_img]
            try:
                position = get_face_position(fn)
            except Exception as err:
                print(err)
                continue
            if len(position) > 0:
                positions[i_img] = position
                set_fn.add(fn)
            bar.update()
        face_position_data.append(positions)

    json.dump(face_position_data, open(f"{args.dataset}.faceBB", "w"))

    with open("_list_of_processed_image_files.log", "w") as h:
        for fn in set_fn:
            h.write(f"{fn}\n")
