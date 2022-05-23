import os
from datetime import datetime
import clip
import torch
from PIL import Image
from torchvision.transforms import functional as TF
import wandb
from torchvision import transforms
from cutouts import MakeCutouts
from prompt import fetch, parse_prompt
from loss import spherical_dist_loss, tv_loss, range_loss

from tqdm.notebook import tqdm

SAVE_N = 10


def do_run(device, models, model_config, settings):
    if settings["seed"] is not None:
        torch.manual_seed(settings["seed"])

    clip_size = models["clip_model"].visual.input_resolution

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    make_cutouts = MakeCutouts(clip_size, settings["cutn"])
    side_x = side_y = model_config["image_size"]

    target_embeds, weights = [], []

    for prompt in settings["prompts"]:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(
            models["clip_model"].encode_text(clip.tokenize(txt).to(device)).float()
        )
        weights.append(weight)

    for prompt in settings["image_prompts"]:
        path, weight = parse_prompt(prompt)
        img = Image.open(fetch(path)).convert("RGB")
        img = TF.resize(
            img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS
        )
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = models["clip_model"].encode_image(normalize(batch)).float()
        target_embeds.append(embed)
        weights.extend([weight / settings["cutn"]] * settings["cutn"])

    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError("The weights must not sum to 0.")
    weights /= weights.sum().abs()

    init = None
    if settings["init_image"] is not None:
        init = Image.open(fetch(settings["init_image"])).convert("RGB")
        init = init.resize((side_x, side_y), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    cur_t = None

    def cond_fn(x, t, out, y=None):
        n = x.shape[0]
        fac = models["diffusion"].sqrt_one_minus_alphas_cumprod[cur_t]
        x_in = out["pred_xstart"] * fac + x * (1 - fac)
        clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
        image_embeds = models["clip_model"].encode_image(clip_in).float()
        dists = spherical_dist_loss(
            image_embeds.unsqueeze(1), target_embeds.unsqueeze(0)
        )
        dists = dists.view([settings["cutn"], n, -1])
        losses = dists.mul(weights).sum(2).mean(0)
        tv_losses = tv_loss(x_in)
        range_losses = range_loss(out["pred_xstart"])
        loss = (
            losses.sum() * settings["clip_guidance_scale"]
            + tv_losses.sum() * settings["tv_scale"]
            + range_losses.sum() * settings["range_scale"]
        )
        if init is not None and settings["init_scale"]:
            init_losses = models["lpips_model"](x_in, init)
            loss = loss + init_losses.sum() * settings["init_scale"]
        return -torch.autograd.grad(loss, x)[0]

    if model_config["timestep_respacing"].startswith("ddim"):
        sample_fn = models["diffusion"].ddim_sample_loop_progressive
    else:
        sample_fn = models["diffusion"].p_sample_loop_progressive

    for i in range(settings["n_batches"]):
        cur_t = models["diffusion"].num_timesteps - settings["skip_timesteps"] - 1

        samples = sample_fn(
            models["model"],
            (settings["batch_size"], 3, side_y, side_x),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=settings["skip_timesteps"],
            init_image=init,
            randomize_class=True,
            cond_fn_with_grad=True,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % SAVE_N == 0 or cur_t == -1:
                print()
                for k, image in enumerate(sample["pred_xstart"]):
                    save_and_log(image,
                                 step=j,
                                 batch=i,
                                 k_img=k,
                                 batch_name=settings["batch_name"],
                                 batch_size=settings["batch_size"],
                                 n_batches=settings["n_batches"])


def save_and_log(image, step=0, batch=1, k_img=1, batch_name="LucidRun", batch_size=None, n_batches=1):
    if n_batches > 1:
        batch_folder = f"{batch_name}/batch_{batch * batch_size + k_img:03}"
    else:
        batch_folder = f"{batch_name}"
    current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
    filename = f'{batch_folder}/{current_time}.png'
    if not os.path.isdir(batch_folder):
        os.makedirs(batch_folder)
    img_0 = image.add(1).div(2).clamp(0, 1)
    image = TF.to_pil_image(img_0).save(filename)
    log_image = wandb.Image(img_0, caption=f"{step}")
    log = {"image": log_image}
    wandb.log(step=step, data=log)
    tqdm.write(f"Batch {batch}, step {step}, output {k_img}:")
    # display.display(display.Image(filename))

