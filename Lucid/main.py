import gc
import io
import logging
import math
import sys
from types import SimpleNamespace

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

import run_loop
import wandb
from cutouts import MakeCutouts
from loss import range_loss, spherical_dist_loss, tv_loss
from model import get_model_config, get_models
import naming

LOGGER = logging.getLogger(__name__)


def set_size(args):
    args.side_x = (args.width_height[0]//64)*64;
    args.side_y = (args.width_height[1]//64)*64;
    if args.side_x != args.width_height[0] or args.side_y != args.width_height[1]:
        print(f'Changing output size to {args.side_x}x{args.side_y}. Dimensions must by multiples of 64.')


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_size(args)
    naming.create_batch_folders(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.model_config = get_model_config(args)
    models = get_models(args, device)
    gc.collect()

    LOGGER.info(args)

    wandb.init(project="LucidEngine", entity="sungam")

    run_loop.do_run(
        args,
        device = device,
        models = models,
    )


if __name__ == "__main__":
    run()
