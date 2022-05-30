"""Namming module"""
import os
from datetime import datetime
from pathlib import Path


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_filename(args, batch, step, k_img):
    current_time = datetime.now().strftime("%y%m%d-%H%M%S_%f")
    filename_template = f"{args.env.root_path}/{args.env.OUTPUT_DIR}/{args.batch_name}/{batch}/{k_img}/{current_time}_{step}{args.env.IMG_EXT}"
    return Path(eval(f"f'{filename_template}'"))

