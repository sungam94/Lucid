import os
import sys

pathlist = ["./CLIP/", "./guided-diffusion/", "./ResizeRight"]
os.environ["PATH"] += os.pathsep + os.pathsep.join(pathlist)
for path in pathlist:
    print(os.path.abspath(path))
    sys.path.append(os.path.abspath(path))

import gc
import logging
import tempfile

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from Lucid import LucidModels, naming, run_loop

LOGGER = logging.getLogger(__name__)


def set_size(args):
    args.side_x = (args.width_height[0] // 64) * 64
    args.side_y = (args.width_height[1] // 64) * 64
    if args.side_x != args.width_height[0] or args.side_y != args.width_height[1]:
        print(
            f"Changing output size to {args.side_x}x{args.side_y}. Dimensions must by multiples of 64."
        )


def cfg2dict(cfg: DictConfig):
    """
    Recursively convert OmegaConf to vanilla dict
    :param cfg:
    :return:
    """
    cfg_dict = {}
    for k, v in cfg.items():
        if type(v) == DictConfig:
            cfg_dict[k] = cfg2dict(v)
        else:
            cfg_dict[k] = v
    return cfg_dict


@hydra.main(version_base=None, config_path="Lucid/configs", config_name="config")
def run(args: DictConfig):
    set_size(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # run = wandb.init(entity=args.wandb.entity, project=args.wandb.project)
    wandb.init(
        # settings=wandb.Settings(start_method="thread﻿"),
        project="LucidEngine",
        entity="sungam",
        mode="online",
    )
    # wandb.log(step=0, data=OmegaConf.to_container(args, resolve=[True | False]))

    models = LucidModels.get_models(args, device)
    gc.collect()
    torch.cuda.empty_cache()


    for key, _model in models.items():
        if key == "diffusion":
            continue
        wandb.watch(_model, log_freq=10)

    LOGGER.info(OmegaConf.to_yaml(args))
    run_loop.do_run(
        args, device=device, models=models,
    )


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    finally:
        gc.collect()
        torch.cuda.empty_cache()

