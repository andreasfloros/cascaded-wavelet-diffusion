import json
import math
import modules
import os
import time
import torch as th
import torchvision as tv
from typing import Optional
import utils
import uuid


@th.inference_mode()
def main(checkpoint_path: str,
         config_path: str,
         out_path: str,
         num_samples: int,
         batch_size: int,
         sample_cfg_path: str,
         seed: Optional[int]) -> None:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if seed is not None:
        th.manual_seed(seed)

    with open(config_path, "r") as f:
        config = json.load(f)
    model = modules.CascadedWaveletDiffuser(num_channels=config["num_channels"],
                                            wd_cfgs=config["levels"]).eval().to(device)
    with open(sample_cfg_path, "r") as f:
        sample_cfg = json.load(f)
    if sample_cfg["ema"]:
        ema = th.optim.swa_utils.AveragedModel(model,
                                               multi_avg_fn=th.optim.swa_utils.get_ema_multi_avg_fn(
                                                   config["ema_decay"]))
        utils.load_state(checkpoint_path=checkpoint_path, model=model, ema=ema)
        for wd, ema_wd in zip(model.wds, ema.module.wds):
            ema_wd.transform = wd.transform
        model = ema.module
    else:
        utils.load_state(checkpoint_path=checkpoint_path, model=model)

    image_size = config["image_size"]
    scale = 2 ** len(model.wds)
    coarse_shape = [image_size[0] // scale, image_size[1] // scale]
    num_batches = math.ceil(num_samples / batch_size)
    start = time.perf_counter()
    for _ in range(num_batches):
        imgs = th.randn(batch_size, config["num_channels"], *coarse_shape, device=device)
        imgs = model.sample(imgs, sample_cfg["levels"])
        for img in imgs:
            tv.utils.save_image(img,
                                os.path.join(out_path, f"{uuid.uuid4()}.png"),
                                normalize=True,
                                value_range=(-1, 1))
    print(f"Done in {time.perf_counter() - start:.2f} seconds.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1, help="number of samples")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="path to config")
    parser.add_argument("--out_path", type=str, required=True, help="path to output images")
    parser.add_argument("--sample_config_path", type=str, required=True, help="path to sample config")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    main(checkpoint_path=args.checkpoint_path,
         config_path=args.config_path,
         out_path=args.out_path,
         num_samples=args.num_samples,
         batch_size=args.batch_size,
         sample_cfg_path=args.sample_config_path,
         seed=args.seed)
