# Imports

import gc
import io
import math
import sys

import wandb
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

import run
from cutouts import MakeCutouts
from loss import range_loss, spherical_dist_loss, tv_loss
from prompt import parse_prompt
from model import get_models, get_model_config

# self.Model settings
class LucidEngine:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_config = get_model_config()
        self.models = get_models(self.device, self.model_config)

        self.run_settings = {}
        self.run_settings["batch_name"] = "mandala"
        self.run_settings["prompts"] = ["psychedelic mandala"]
        self.run_settings["image_prompts"] = []
        self.run_settings["batch_size"] = 1
        self.run_settings["clip_guidance_scale"] = (
            1000  # Controls how much the image should look like the prompt.
        )
        self.run_settings["tv_scale"] = 500  # Controls the smoothness of the final output.
        self.run_settings["range_scale"] = 50  # Controls how far out of range RGB values are allowed to be.
        self.run_settings["cutn"] = 16
        self.run_settings["n_batches"] = 1
        self.run_settings["init_image"] = (
            None  # This can be an URL or Colab local path and must be in quotes.
        )
        self.run_settings["skip_timesteps"] = (
            0  # This needs to be between approx. 200 and 500 when using an init image.
        )
        # Higher values make the output look more like the init.
        self.run_settings["init_scale"] = (
            0  # This enhances the effect of the init image, a good value is 1000.
        )
        self.run_settings["seed"] = 0

        gc.collect()

    def do_run(self):
        print(self.models.keys)
        wandb.log(data=self.run_settings, step=0)


        run.do_run(
            device=self.device,
            models=self.models,
            model_config=self.model_config,
            settings=self.run_settings
        )


if __name__ == "__main__":
    wandb.init(project="LucidEngine", entity="sungam")
    Engine = LucidEngine()
    Engine.do_run()
