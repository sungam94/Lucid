import gc
import logging
import random
from datetime import datetime

import clip
import torch
from omegaconf import open_dict
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

import wandb
from Lucid import naming
from Lucid.cutouts import MakeCutouts
from Lucid.loss import range_loss, spherical_dist_loss, tv_loss
from Lucid.prompt import fetch, parse_prompt
from Lucid.cutouts import MakeCutoutsDango
LOGGER = logging.getLogger(__name__)
SAVE_N = 10


def write_step_to_args(args, batch, step, k_img):
    """ Update args with current step indeces

    Write batch number, step and, k_img, and time into args,
    so the templates in maing can pick it up
    """
    current_time = datetime.now().strftime("%y%m%d-%H%M%S_%f")

    with open_dict(args):
        args["batch_{batch}"] = {}
        args["batch_{batch}"].time = current_time
        args["batch_{batch}"].index = batch
        args["batch_{batch}"].step = step
        args["batch_{batch}"].k_img = k_img


def save_and_log(args, image, batch=None, step=None, k_img=None):
    img_0 = image.add(1).div(2).clamp(0, 1)
    log_image = wandb.Image(img_0, caption=f"{step}")
    log = {
        f"batch_{batch * args.batch_size + k_img:03}": log_image,
        "batch": batch,
        "k_img": k_img,
    }
    print(f"step: {step}")
    wandb.log(step=step, data=log)

    filename = naming.get_filename(args, batch, step, k_img)
    naming.create_directory(filename.parent)
    image = TF.to_pil_image(img_0).save(filename)

    tqdm.write(f"Batch {batch}, step {step}, output {k_img}:")


def get_prompt_embeddings(args, clip_model, device=None):
    """
    creation of image vectors.
    weight are the relative prompt weights.
    """
    target_embeds, weights = [], []
    for prompt in args.prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(
            clip_model.encode_text(clip.tokenize(txt).to(device)).float()
        )
        weights.append(weight)

    # for prompt in args.image_prompts:
    #     path, weight = parse_prompt(prompt)
    #     img = Image.open(fetch(path)).convert("RGB")
    #     img = TF.resize(
    #         img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS
    #     )
    #     batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
    #     embed = models["clip_model"].encode_image(normalize(batch)).float()
    #     target_embeds.append(embed)
    #     weights.extend([weight / args.cutn] * args.cutn)

    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError("The weights must not sum to 0.")
    weights /= weights.sum().abs()
    return target_embeds, weights


def get_safe_size(args):
    side_x = (args.width_height[0] // 64) * 64
    side_y = (args.width_height[1] // 64) * 64
    if side_x != args.width_height[0] or side_y != args.width_height[1]:
        print(
            f"Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64."
        )
    return side_x, side_y


def set_seed(args):
    if args.seed == "random_seed":
        random.seed()
        seed = random.randint(0, 2 ** 32)
        print(f"Using seed: {seed}")
    else:
        seed = int(args.seed)
    # np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed


def do_run(args, device, models):
    torch.cuda.empty_cache()
    set_seed(args)
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    clip_size = models["clip_model"].visual.input_resolution
    make_cutouts = MakeCutouts(clip_size, args.cutn)

    side_x, side_y = get_safe_size(args)

    prompt_embeds, prompt_weights = get_prompt_embeddings(
        args, models["clip_model"], device=device
    )
    init = None
    # if args.init_image is not None:
    #     init = Image.open(fetch(args.init_image)).convert("RGB")
    #     init = init.resize((side_x, side_y), Image.LANCZOS)
    #     init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

    cur_t = None

    # define conditioning functionwith losses
    def cond_fn(x, t, y=None):
        """

        Args:
            t: float, scaled between 1000 and num_steps?
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            out = models["diffusion"].p_mean_variance(
                models["model"], x, my_t, clip_denoised=False, model_kwargs={"y": y}
            )
            fac = models["diffusion"].sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out["pred_xstart"] * fac + x * (1 - fac)

            x_in_grad = torch.zeros_like(x_in)

            # clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
            # image_embeds = models["clip_model"].encode_image(normalize(clip_in)).float()
            # img_in = make_cutouts(TF.to_tensor(x_in).to(device).unsqueeze(0).mul(2).sub(1))

            t_int = int(t.item())+1 #errors on last step without +1, need to find source

            cuts = MakeCutoutsDango(
                args,
                clip_size,
                Overview=eval(args.cut_overview)[1000 - t_int],
                InnerCrop=eval(args.cut_innercut)[1000 - t_int],
                IC_Size_Pow=args.cut_ic_pow,
                IC_Grey_P=eval(args.cut_icgray_p)[1000 - t_int],
            )
            clip_in = normalize(cuts(x_in.add(1).div(2)))
            image_embeds = models["clip_model"].encode_image(clip_in).float()

            # LOSSES
            dists = spherical_dist_loss(
                image_embeds.unsqueeze(1), prompt_embeds.unsqueeze(0)
            )
            dists = dists.view([args.cutn, n, -1])
            losses = dists.mul(prompt_weights).sum(2).mean(0)
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out["pred_xstart"])
            loss = (
                losses.sum() * args.clip_guidance_scale
                + tv_losses.sum() * args.tv_scale
                + range_losses.sum() * args.range_scale
            )

            x_in_grad += (
                torch.autograd.grad(losses.sum() * args.clip_guidance_scale, x_in)[0]
                / args.cutn_batches
            )
            # wandb.log(
            #     step=args.steps - cur_t - 1,
            #     data={
            #         "step": t,
            #         "tv_loss": tv_losses,
            #         "range_loss": range_losses,
            #         "loss": loss,
            #     },
            # )

            if init is not None and args.init_scale:
                init_losses = models["lpips_model"](x_in, init)
                loss = loss + init_losses.sum() * args.init_scale
            if torch.isnan(x_in_grad).any() == False:
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                # print("NaN'd")
                x_is_NaN = True
                grad = torch.zeros_like(x)

        return grad
        # return -torch.autograd.grad(loss, x)[0]

    # get sample generator
    if args.model_config.timestep_respacing.startswith("ddim"):
        sample_fn = models["diffusion"].ddim_sample_loop_progressive
    else:
        sample_fn = models["diffusion"].plms_sample_loop_progressive

    for i in range(args.n_batches):
        cur_t = models["diffusion"].num_timesteps - args.skip_timesteps - 1

        samples = sample_fn(
            models["model"],
            (args.batch_size, 3, side_y, side_x),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=args.skip_timesteps,
            init_image=init,
            randomize_class=False,
            # order=2,
            # cond_fn_with_grad=True,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % SAVE_N == 0 or cur_t == -1:
                for k, image in enumerate(sample["pred_xstart"]):
                    print(f"step: {j}, batch: {i}, k_img: {k}")
                    save_and_log(args, image, batch=i, step=j, k_img=k)
                    gc.collect()
                    torch.cuda.empty_cache()
