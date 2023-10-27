import numpy as np
import torch


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def denoising_step(xt, c, t, t_next, *,
                   model,
                   b,
                   eta=0.0,
                   unconditional_guidance_scale=1., 
                   unconditional_conditioning=None,
                   att_mask = None
                   ):
    
    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        et = model.apply_model(xt, t, c, att_mask=att_mask)
    elif att_mask is not None:
        x_in = xt
        t_in = t
        c_in = unconditional_conditioning
        et_uncond = model.apply_model(x_in, t_in, c_in)
        c_in = c
        et = model.apply_model(x_in, t_in, c_in, att_mask=att_mask)
        et = et_uncond + unconditional_guidance_scale * (et - et_uncond)
    else:
        x_in = torch.cat([xt] * 2)
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([unconditional_conditioning, c])
        att_mask = None if att_mask is None else torch.cat([att_mask, att_mask])
        et_uncond, et = model.apply_model(x_in, t_in, c_in, att_mask=att_mask).chunk(2)
        et = et_uncond + unconditional_guidance_scale * (et - et_uncond)
    
    # Compute the next x
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)
    
    if t_next.sum() == -t_next.shape[0]:
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)
    
    xt_next = torch.zeros_like(xt)
    
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    if eta == 0:
        xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    else:
        c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)
    
    return xt_next, x0_t


