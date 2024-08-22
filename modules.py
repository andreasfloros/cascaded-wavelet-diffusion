import json
import improved_diffusion
import torch as th
from torch.utils.data import DataLoader
import torchvision as tv
from typing import Any, Dict, List, Optional, Tuple
import utils


class hwt2d(th.nn.Module):
    """
    Haar transform.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()

        self.register_buffer("details_mean", th.zeros(1, 3 * num_channels, 1, 1))
        self.register_buffer("details_var", th.ones(1, 3 * num_channels, 1, 1))
        self.register_buffer("details_min", th.empty(1, 3 * num_channels, 1, 1))
        self.register_buffer("details_max", th.empty(1, 3 * num_channels, 1, 1))
        self.details_min.copy_(-th.inf)
        self.details_max.copy_(th.inf)

    def forward(self, inpt: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return utils.haar2d(inpt)

    inverse = staticmethod(utils.ihaar2d)


class Diffuser(th.nn.Module):
    """
    VP diffuser module.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 T: int,
                 linear: bool,
                 early_timestep_sharing: bool,
                 unet_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.early_timestep_sharing = early_timestep_sharing
        self.unet = improved_diffusion.UNetModel(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 channel_mult=[1,] + unet_cfg["channel_mults"],
                                                 model_channels=unet_cfg["base_channels"],
                                                 num_res_blocks=unet_cfg["num_res_attn_blocks"],
                                                 attention_resolutions=[2 ** i for i, is_attn in enumerate(unet_cfg["is_attn"]) if is_attn],  # noqa: E501
                                                 dropout=unet_cfg["dropout"],
                                                 num_heads=unet_cfg["num_heads"],
                                                 use_scale_shift_norm=unet_cfg["use_scale_shift_norm"],
                                                 )
        if linear:
            betas = th.linspace(0.1 / T, 20 / T, T, dtype=th.float64)
        else:
            s = 0.008
            steps = th.linspace(0., T, T + 1, dtype=th.float64)
            ft = th.cos(((steps / T + s) / (1 + s)) * th.pi * 0.5) ** 2
            betas = th.clip(1 - ft[1:] / ft[:T], 0., 0.999)

        sqrt_betas = th.sqrt(betas)
        alphas = 1 - betas
        alphas_cumprod = th.cumprod(alphas, dim=0)
        one_minus_alphas_cumprod = 1 - alphas_cumprod
        sqrt_alphas = th.sqrt(alphas)

        sqrt_alphas_cumprod = th.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = th.sqrt(one_minus_alphas_cumprod)

        self.register_buffer("betas", betas.to(th.float32))
        self.register_buffer("sqrt_betas", sqrt_betas.to(th.float32))
        self.register_buffer("alphas", alphas.to(th.float32))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(th.float32))
        self.register_buffer("one_minus_alphas_cumprod", one_minus_alphas_cumprod.to(th.float32))
        self.register_buffer("sqrt_alphas", sqrt_alphas.to(th.float32))
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod.to(th.float32))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod.to(th.float32))

        T = th.tensor(T, dtype=th.float32).unsqueeze_(0)
        self.register_buffer("T", T)

    def forward(self, x: th.Tensor, t: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Noising from 0 to t.
        """

        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        eps = th.randn_like(x)
        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * eps, eps

    def randint(self, batch_size: int, device: th.device) -> th.Tensor:
        """
        Sample a random time step.
        """

        return th.randint(low=0, high=len(self.betas), size=(batch_size, 1, 1, 1), device=device)

    def rescale_time(self, t):
        """
        Rescale time.
        """

        T = len(self.betas)

        # https://openreview.net/pdf?id=WNkW0cOwiz
        if self.early_timestep_sharing:
            interval = 0.1 * T
            small_interval = interval / 5
            t = th.where(t > interval, t, th.floor(t / small_interval) * small_interval)

        return t * 1000. / T

    def epsilon(self, x: th.Tensor, t: th.Tensor) -> th.Tensor:
        return self.unet(x, self.rescale_time(t))


class WaveletDiffuser(th.nn.Module):
    """
    Wavelet diffuser module (basic cascade or super resolver).
    """

    def __init__(self,
                 num_channels: int,
                 early_timestep_sharing: bool,
                 dnc_cfg: Optional[Dict[str, Any]],
                 dnd_cfg: Dict[str, Any]) -> None:
        """
        Initialize the model with configs. Note that the configs are consumed.

        Args:
            num_channels: the number of channels in the model.
            early_timestep_sharing: whether to use early timestep.
            dnc_cfg: the coarse diffuser config.
            dnd_cfg: the details diffuser config.
        """

        super().__init__()
        self.transform = hwt2d(num_channels=num_channels)
        self.dnc = Diffuser(in_channels=num_channels,
                            out_channels=num_channels,
                            early_timestep_sharing=early_timestep_sharing,
                            unet_cfg=dnc_cfg.pop("unet"),
                            **dnc_cfg) if dnc_cfg is not None else None
        self.dnd = Diffuser(in_channels=4 * num_channels,
                            out_channels=3 * num_channels,
                            early_timestep_sharing=early_timestep_sharing,
                            unet_cfg=dnd_cfg.pop("unet"),
                            **dnd_cfg)

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        DDP training.

        Args:
            x: the input tensor.
        Returns:
            coarse, diffuser_loss
        """

        coarse, details = self.transform(x)
        diffuser_loss = 0

        t = self.dnd.randint(x.shape[0], x.device)
        w = 1
        details = (details - self.transform.details_mean) / self.transform.details_var.sqrt()
        details_t, eps = self.dnd(details, t)
        diffuser_loss = (w * (self.dnd.epsilon(
            th.cat([details_t, coarse], dim=1), t) - eps) ** 2).mean()

        if self.dnc is not None:
            t = self.dnc.randint(x.shape[0], x.device)
            w = 1
            coarse_t, eps = self.dnc(coarse, t)
            diffuser_loss = diffuser_loss + (w * (self.dnc.epsilon(coarse_t, t) - eps) ** 2).mean()

        return coarse, diffuser_loss

    @th.inference_mode()
    def sample(self,
               init: th.Tensor,
               cfg: Dict[str, Any]) -> th.Tensor:
        """
        Sample from the model.

        Args:
            init: the initial coarse tensor.
            cfg: the sample config.
        """

        coarse = utils.ddim(self.dnc,
                            init,
                            clamp_min=-1.,
                            clamp_max=1.,
                            steps=cfg.get("coarse_steps", None),
                            eta=cfg.get("coarse_eta", 1.)) if self.dnc is not None else init
        details = utils.ddim(self.dnd,
                             th.randn([coarse.shape[0],
                                      3 * coarse.shape[1],
                                      *coarse.shape[2:]], device=coarse.device),
                             condition=coarse,
                             steps=cfg.get("details_steps", None),
                             eta=cfg.get("details_eta", 1.),
                             clamp_min=(self.transform.details_min - self.transform.details_mean)
                             / self.transform.details_var.sqrt(),
                             clamp_max=(self.transform.details_max - self.transform.details_mean)
                             / self.transform.details_var.sqrt())
        details = details * self.transform.details_var.sqrt() + self.transform.details_mean
        return self.transform.inverse(coarse, details).clamp(-1, 1)


class CascadedWaveletDiffuser(th.nn.Module):
    """
    Cascaded wavelet diffuser module.
    """

    def __init__(self,
                 num_channels: int,
                 early_timestep_sharing: bool,
                 wd_cfgs: List[str]) -> None:
        """
        Initialize the model with configs.

        Args:
            num_channels: the number of channels in the model.
            early_timestep_sharing: whether to use early timestep.
            wd_cfgs: the wavelet diffuser config paths.
        """

        super().__init__()
        self.wds = th.nn.ModuleList()
        for wd_cfg in wd_cfgs:
            with open(wd_cfg, "r") as f:
                config = json.load(f)
            self.wds.append(WaveletDiffuser(num_channels=num_channels,
                                            early_timestep_sharing=early_timestep_sharing,
                                            dnc_cfg=config.get("coarse_diffuser", None),
                                            dnd_cfg=config["details_diffuser"]))

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        DDP training.

        Args:
            x: the input tensor.

        Returns:
            diffuser_loss
        """

        diffuser_loss = 0
        for wd in self.wds:
            x, diffuser_loss_scale = wd(x)
            diffuser_loss = diffuser_loss + diffuser_loss_scale
        return diffuser_loss

    @th.inference_mode()
    def sample(self,
               init: th.Tensor,
               sample_cfg: List[Dict[str, Any]]) -> th.Tensor:
        """
        Sample from the model.

        Args:
            init: the initial coarse tensor.
            sample_cfg: the sample config (ascending order).
        """

        x = init
        for wd, cfg in zip(reversed(self.wds), sample_cfg):
            x = wd.sample(x, cfg)
        return x

    @th.no_grad()
    def calculate_stats(self, data_path: str, batch_size: int = 1, num_workers: int = 0) -> None:
        """
        Calculate the normalization stats for the transforms.

        Args:
            data_path: the path to the data.
            batch_size: the batch size.
            num_workers: the number of workers.
        """

        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        dataset = tv.datasets.ImageFolder(data_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        for wd in self.wds:
            wd.transform.details_mean.zero_()
            wd.transform.details_var.zero_()
            wd.transform.details_min.copy_(th.inf)
            wd.transform.details_max.copy_(-th.inf)
        device = next(self.parameters()).device

        num = len(dataset)

        for inpt, _ in dataloader:
            inpt = inpt.to(device)
            for wd in self.wds:
                inpt, details = wd.transform(inpt)
                wd.transform.details_mean += inpt.shape[0] * details.mean(dim=(0, 2, 3), keepdim=True) / num
                wd.transform.details_min.copy_(th.min(wd.transform.details_min, details.min()))
                wd.transform.details_max.copy_(th.max(wd.transform.details_max, details.max()))

        for inpt, _ in dataloader:
            inpt = inpt.to(device)
            for wd in self.wds:
                inpt, details = wd.transform(inpt)
                wd.transform.details_var += inpt.shape[0] * (details - wd.transform.details_mean).pow(2).mean(dim=(0, 2, 3), keepdim=True) / (num - 1)  # noqa: E501
