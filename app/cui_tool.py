import sys
import argparse
from importlib import import_module
import json
import os
import shutil

import numpy as np

def command_sim(args, module):
    score = module.get_similarity_result([args.files[0]], [args.files[1]])
    print(score[0])

def command_enroll(args, module):
    # Load list of enroll image files.
    with open(args.enroll_config_fn) as h:
        conf = json.load(h)
    character_names, character_directories = zip(*list(conf.items()))
    character_filenames= []
    for directory in character_directories:
        if not os.path.exists(directory):
            raise Exception(f"Directory do not exists: {directory}")
        character_filenames.append(
            module._get_list_of_all_image_files_directory(directory)
        )
    print("Candidate characters:")
    for char, filenames in zip(character_names, character_filenames):
        assert len(filenames) > 0, f"No enroll image file founded for character: {char}"
        print(f"    {char} ({len(filenames)} images)")

    # Load list of target files.
    target_filenames = module._get_list_of_all_image_files_directory(args.target_dir)
    if len(target_filenames) == 0:
        raise Exception(f"No target file in directory: {args.target_dir}")
    print(f"Found {len(target_filenames)} target images.")
    
    # Compute identification scores for each pair of image and character.
    print("Identifying characters...")
    rslt = module.get_identification_result(character_filenames, target_filenames, args.mode)

    # Copy each image files to corresponding directory based on identified characters.
    print("Copying images...")
    def make_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    for i_target in range(rslt.shape[0]):
        i_char = np.argmax(rslt[i_target])

        img_fn = target_filenames[i_target]
        char = character_names[i_char]

        dir_name = os.path.join(args.result_dir, char)
        make_dir(dir_name)
        shutil.copyfile(img_fn, os.path.join(dir_name, os.path.basename(img_fn)))
    print("Done.")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-dir", type=str)
    parser.add_argument("--plugin-name", type=str, default="main")
    parser.add_argument("--gpu", type=int, default=-1)

    subparsers = parser.add_subparsers()

    parser_sim = subparsers.add_parser("similarity")
    parser_sim.add_argument("--files", type=str, nargs=2)
    parser_sim.set_defaults(handler=command_sim)

    parser_enroll = subparsers.add_parser("identify")
    parser_enroll.add_argument("--enroll-config-fn", type=str)
    parser_enroll.add_argument("--target-dir", type=str)
    parser_enroll.add_argument("--mode", choices=["Avg", "Max"])
    parser_enroll.add_argument("--result-dir", type=str, default="out")
    parser_enroll.set_defaults(handler=command_enroll)


    args = parser.parse_args()

    device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
    print(f"Using device: {device}")

    sys.path.append(args.plugin_dir)
    plugin = import_module(args.plugin_name)

    module = plugin.MainModule(device=device)

    if hasattr(args, "handler"):
        args.handler(args, module)
    else:
        parser.print_help()