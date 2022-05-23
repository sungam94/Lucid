import torch
import clip

import lpips
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

def get_model_config():
    model_config = model_and_diffusion_defaults()
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": "100",  # Modify this value to decrease the number of
            # timesteps.
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": False,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )
    return model_config


def get_models(device, model_config):
    print("Using device:", device)

    model, diffusion = create_model_and_diffusion(**model_config)

    model.load_state_dict(
        torch.load("./models/256x256_diffusion_uncond.pt", map_location="cpu")
    )

    model.requires_grad_(False).eval().to(device)
    if model_config["use_fp16"]:
        model.convert_to_fp16()



    clip_model = (
        clip.load("ViT-B/16", jit=False)[0]
        .eval()
        .requires_grad_(False)
        .to(device)
    )



    lpips_model = lpips.LPIPS(net="vgg").to(device)



    models = {
        "model": model,
        "diffusion": diffusion,
        "clip_model": clip_model,
        "lpips+model": lpips_model,
    }
    return models
