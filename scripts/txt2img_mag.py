"""make variations of input image"""

import argparse, os, sys, glob
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import PIL
import cv2
import torch

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def main():
        
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="A furry bear riding on a bike in the city",
        help="the prompt to render"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        nargs="+",
        default="low quality",
        help="the negative prompt"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )

    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
        default="checkpoint/512-base-ema.ckpt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--mask",
        type=str,
        help="path to semantic masks",
        default="inputs/mask1.png"
    )
    parser.add_argument(
        "--word_ids_for_mask",
        type=str,
        help="corresponding id list for each mask",
        default= '[[1,2,3,4],[6,7]]'
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="masked attention guidance scale",
        default= 0.08
    )
    parser.add_argument(
        "--lmda",
        type=float,
        help="masked attention loss balancing weight",
        default= 0.5
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    
    word_ids_for_mask = eval(opt.word_ids_for_mask)
    mask_path = opt.mask
    alpha = opt.alpha
    lmbd = opt.lmda

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    
    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
    
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))
            
    negative_prompt = [opt.negative_prompt]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    init_image = torch.zeros((batch_size,3,512,512)).to(device)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space    
    
    mask_img = cv2.resize(cv2.imread(mask_path),init_latent.shape[2:],interpolation=cv2.INTER_NEAREST)
    mask_img = mask_img.reshape((mask_img.shape[0]*mask_img.shape[1], 3)).tolist()
    mask_img = [str(s) for s in mask_img]
    mask_colors = list(set(mask_img))
    mask_colors.sort(reverse=True)
    print("Order of regions:", mask_colors)
    print("Specified word IDs for each region:", word_ids_for_mask)

    mask_img = np.array(mask_img).reshape(init_latent.shape[2:])
    att_masks = []
    for mask_color in mask_colors:
        mask = torch.zeros((init_latent.shape[0],init_latent.shape[2],init_latent.shape[3]), device=init_latent.device)
        for b_i in range(init_latent.shape[0]):
            mask[b_i][mask_img==mask_color]=1.
        att_masks.append(mask.unsqueeze(1))

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")
    
    
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * negative_prompt)
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    c = model.get_learned_conditioning(prompts)

                    samples = torch.randn(init_latent.shape, device=device)
                    #"==Denoising with masked-attention swapping=="
                    #samples = sampler.attention_guided_reverse_ddim(samples, c, unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc, 
                    #                                                att_masks=att_masks, word_ids_for_mask=word_ids_for_mask, alpha=alpha, lmbd=lmbd, swapping_step_th=700)
                    "==Denoising with masked-attention guidance=="
                    samples = sampler.attention_guided_reverse_ddim(samples, c, unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc, 
                                                                    att_masks=att_masks, word_ids_for_mask=word_ids_for_mask, alpha=alpha, lmbd=lmbd, guidance_step_th=700)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1
                    all_samples.append(x_samples)

            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            grid = put_watermark(grid, wm_encoder)
            grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")


if __name__ == "__main__":
    main()
