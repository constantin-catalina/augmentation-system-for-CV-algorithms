import argparse
import os
import sys

import cv2
import numpy as np
import tkinter as tk
    
from tkinter import filedialog
from config_parser import ConfigError, load_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "config", "valid_spec_example.txt")


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, img):
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError as e:
        print(f"  {e}")
        return False
    return True


def select_directory():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial = os.path.join(BASE_DIR, "test")
    path = filedialog.askdirectory(
        title="Select the directory with .jpg images",
        initialdir=initial if os.path.isdir(initial) else BASE_DIR,
    )
    root.destroy()
    return path


def list_jpgs(directory):
    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(directory, f))
    )


def apply_chain(img, chain):
    """Apply the operations of a chain in order."""
    for op in chain:
        img = op.func(img, *op.args)
    return img


def chain_label(chain):
    return "-".join(op.name for op in chain)


def run(in_dir, steps, clean=False, verbose=True):
    in_dir = os.path.normpath(in_dir)
    files = list_jpgs(in_dir)
    out_dir = in_dir + "_aug"
    os.makedirs(out_dir, exist_ok=True)

    if clean:
        for old in list_jpgs(out_dir):
            os.remove(os.path.join(out_dir, old))
    elif os.listdir(out_dir) and verbose:
        print(f"Note: {out_dir} is not empty; files with the same name are overwritten "
              f"(use --clean to delete the old .jpg files first).")

    counter = 0
    unreadable = set()
    for chain in steps:
        label = chain_label(chain)
        for fname in files:
            img = read_image(os.path.join(in_dir, fname))
            if img is None:
                if verbose and fname not in unreadable:
                    print(f"  skipped (cannot read): {fname}")
                unreadable.add(fname)
                continue
            result = apply_chain(img, chain)

            counter += 1
            out_name = f"{os.path.splitext(fname)[0]}_{label}_{counter}.jpg"
            if write_image(os.path.join(out_dir, out_name), result):
                if verbose:
                    print(f"  saved {out_name}")
            else:
                print(f"  FAILED to save {out_name}")
                counter -= 1
    return out_dir, counter


def main(argv=None):
    parser = argparse.ArgumentParser(description="Image augmentation system.")
    parser.add_argument("directory", nargs="?",
                        help="directory with .jpg images (if missing, a dialog opens)")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG,
                        help="configuration file (default: config/valid_spec_example.txt)")
    parser.add_argument("--clean", action="store_true",
                        help="delete the old .jpg files from the output directory first")
    args = parser.parse_args(argv)

    try:
        steps = load_config(args.config)
    except (OSError, ConfigError) as e:
        sys.exit(f"Config error: {e}")
    print(f"Loaded {len(steps)} augmentation step(s) from {args.config}")

    in_dir = args.directory or select_directory()
    if not in_dir:
        sys.exit("No directory selected.")
    if not os.path.isdir(in_dir):
        sys.exit(f"Not a directory: {in_dir}")
    if not list_jpgs(in_dir):
        sys.exit(f"No .jpg images found in {in_dir}")

    out_dir, count = run(in_dir, steps, clean=args.clean)
    print(f"Done. {count} image(s) saved in {out_dir}")


if __name__ == "__main__":
    main()
