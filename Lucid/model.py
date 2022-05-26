import hashlib
import os

from omegaconf import DictConfig, OmegaConf
import hydra
import clip
import lpips
import torch
from guided_diffusion.script_util import (create_model_and_diffusion,
                                          model_and_diffusion_defaults)


def get_model_config(args):
    clip_model = args.models.model
    model_config = model_and_diffusion_defaults()
    model_config.update(args.model_settings.get(clip_model))

    return model_config


def get_models(args, device):
    model, diffusion = create_model_and_diffusion(**args.model_config)
    model.load_state_dict(
        torch.load(f"{args.env.ROOT_DIR}/models/{args.models.model}.pt", map_location="cpu")
    )
    model.requires_grad_(False).eval().to(device)
    if args.model_config.use_fp16:
        model.convert_to_fp16()

    clip_model = (
        clip.load(args.models.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    )
    lpips_model = lpips.LPIPS(net="vgg").to(device)
    models = {
        "model": model,
        "diffusion": diffusion,
        "clip_model": clip_model,
        "lpips+model": lpips_model,
    }
    return models


"""
ViTB32 = True  # @param{type:"boolean"}
ViTB16 = True  # @param{type:"boolean"}
ViTL14 = False  # @param{type:"boolean"} # Default False
RN101 = True  # @param{type:"boolean"} # Default False
RN50 = True  # @param{type:"boolean"} # Default True
RN50x4 = True  # @param{type:"boolean"} # Default False
RN50x16 = False  # @param{type:"boolean"}
RN50x64 = False  # @param{type:"boolean"}
SLIPB16 = False  # param{type:"boolean"} # Default False. Looks broken, likely related to commented import of SLIP_VITB16
SLIPL16 = False  # param{type:"boolean"}


# @markdown If you're having issues with model downloads, check this to compare SHA's:
check_model_SHA = False  # @param{type:"boolean"}

model_256_SHA = "983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a"
model_512_SHA = "9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648"
model_secondary_SHA = "983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a"

model_256_link = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
model_512_link = "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt"
model_secondary_link = (
    "https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth"
)

model_256_path = f"{model_path}/256x256_diffusion_uncond.pt"
model_512_path = f"{model_path}/512x512_diffusion_uncond_finetune_008100.pt"
model_secondary_path = f"{model_path}/secondary_model_imagenet_2.pth"

# Download the diffusion model
if diffusion_model == "256x256_diffusion_uncond":
    if os.path.exists(model_256_path) and check_model_SHA:
        print("Checking 256 Diffusion File")
        with open(model_256_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()
        if hash == model_256_SHA:
            print("256 Model SHA matches")
            model_256_downloaded = True
        else:
            print("256 Model SHA doesn't match, redownloading...")
            get_ipython().system("wget --continue {model_256_link} -P {model_path}")
            model_256_downloaded = True
    elif (
        os.path.exists(model_256_path)
        and not check_model_SHA
        or model_256_downloaded == True
    ):
        print(
            "256 Model already downloaded, check check_model_SHA if the file is corrupt"
        )
    else:
        get_ipython().system("wget --continue {model_256_link} -P {model_path}")
        model_256_downloaded = True
elif diffusion_model == "512x512_diffusion_uncond_finetune_008100":
    if os.path.exists(model_512_path) and check_model_SHA:
        print("Checking 512 Diffusion File")
        with open(model_512_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()
        if hash == model_512_SHA:
            print("512 Model SHA matches")
            model_512_downloaded = True
        else:
            print("512 Model SHA doesn't match, redownloading...")
            get_ipython().system("wget --continue {model_512_link} -P {model_path}")
            model_512_downloaded = True
    elif (
        os.path.exists(model_512_path)
        and not check_model_SHA
        or model_512_downloaded == True
    ):
        print(
            "512 Model already downloaded, check check_model_SHA if the file is corrupt"
        )
    else:
        get_ipython().system("wget --continue {model_512_link} -P {model_path}")
        model_512_downloaded = True


# Download the secondary diffusion model v2
if use_secondary_model == True:
    if os.path.exists(model_secondary_path) and check_model_SHA:
        print("Checking Secondary Diffusion File")
        with open(model_secondary_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()
        if hash == model_secondary_SHA:
            print("Secondary Model SHA matches")
            model_secondary_downloaded = True
        else:
            print("Secondary Model SHA doesn't match, redownloading...")
            get_ipython().system(
                "wget --continue {model_secondary_link} -P {model_path}"
            )
            model_secondary_downloaded = True
    elif (
        os.path.exists(model_secondary_path)
        and not check_model_SHA
        or model_secondary_downloaded == True
    ):
        print(
            "Secondary Model already downloaded, check check_model_SHA if the file is corrupt"
        )
    else:
        get_ipython().system("wget --continue {model_secondary_link} -P {model_path}")
        model_secondary_downloaded = True

model_config = model_and_diffusion_defaults()
if diffusion_model == "512x512_diffusion_uncond_finetune_008100":
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": diffusion_steps,
            "rescale_timesteps": True,
            "timestep_respacing": timestep_respacing,
            "image_size": 512,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": use_checkpoint,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )
elif diffusion_model == "256x256_diffusion_uncond":
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": diffusion_steps,
            "rescale_timesteps": True,
            "timestep_respacing": timestep_respacing,
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": use_checkpoint,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )

secondary_model_ver = 2
model_default = model_config["image_size"]


if secondary_model_ver == 2:
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(
        torch.load(f"{model_path}/secondary_model_imagenet_2.pth", map_location="cpu")
    )
secondary_model.eval().requires_grad_(False).to(device)

clip_models = []
if ViTB32 is True:
    clip_models.append(
        clip.load("ViT-B/32", jit=False)[0].eval().requires_grad_(False).to(device)
    )
if ViTB16 is True:
    clip_models.append(
        clip.load("ViT-B/16", jit=False)[0].eval().requires_grad_(False).to(device)
    )
if ViTL14 is True:
    clip_models.append(
        clip.load("ViT-L/14", jit=False)[0].eval().requires_grad_(False).to(device)
    )
if RN50 is True:
    clip_models.append(
        clip.load("RN50", jit=False)[0].eval().requires_grad_(False).to(device)
    )
if RN50x4 is True:
    clip_models.append(
        clip.load("RN50x4", jit=False)[0].eval().requires_grad_(False).to(device)
    )
if RN50x16 is True:
    clip_models.append(
        clip.load("RN50x16", jit=False)[0].eval().requires_grad_(False).to(device)
    )
if RN50x64 is True:
    clip_models.append(
        clip.load("RN50x64", jit=False)[0].eval().requires_grad_(False).to(device)
    )
if RN101 is True:
    clip_models.append(
        clip.load("RN101", jit=False)[0].eval().requires_grad_(False).to(device)
    )

if SLIPB16:
    SLIPB16model = SLIP_VITB16(ssl_mlp_dim=4096, ssl_emb_dim=256)
    if not os.path.exists(f"{model_path}/slip_base_100ep.pt"):
        get_ipython().system(
            "wget https://dl.fbaipublicfiles.com/slip/slip_base_100ep.pt -P {model_path}"
        )
    sd = torch.load(f"{model_path}/slip_base_100ep.pt")
    real_sd = {}
    for k, v in sd["state_dict"].items():
        real_sd[".".join(k.split(".")[1:])] = v
    del sd
    SLIPB16model.load_state_dict(real_sd)
    SLIPB16model.requires_grad_(False).eval().to(device)

    clip_models.append(SLIPB16model)

if SLIPL16:
    SLIPL16model = SLIP_VITL16(ssl_mlp_dim=4096, ssl_emb_dim=256)
    if not os.path.exists(f"{model_path}/slip_large_100ep.pt"):
        get_ipython().system(
            "wget https://dl.fbaipublicfiles.com/slip/slip_large_100ep.pt -P {model_path}"
        )
    sd = torch.load(f"{model_path}/slip_large_100ep.pt")
    real_sd = {}
    for k, v in sd["state_dict"].items():
        real_sd[".".join(k.split(".")[1:])] = v
    del sd
    SLIPL16model.load_state_dict(real_sd)
    SLIPL16model.requires_grad_(False).eval().to(device)

    clip_models.append(SLIPL16model)

normalize = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)
lpips_model = lpips.LPIPS(net="vgg").to(device)

"""
