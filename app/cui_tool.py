import sys
import argparse
from importlib import import_module

def command_sim(args, module):
    score = module.get_similarity_result([args.files[0]], [args.files[1]])
    print(score[0])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-dir", type=str)
    parser.add_argument("--plugin-name", type=str, default="main")
    parser.add_argument("--gpu", type=int, default=-1)

    subparsers = parser.add_subparsers()

    parser_sim = subparsers.add_parser("similarity")
    parser_sim.add_argument("--files", type=str, nargs=2)
    parser_sim.set_defaults(handler=command_sim)

    args = parser.parse_args()

    device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"

    sys.path.append(args.plugin_dir)
    plugin = import_module(args.plugin_name)

    module = plugin.MainModule(device=device)

    if hasattr(args, "handler"):
        args.handler(args, module)
    else:
        parser.print_help()