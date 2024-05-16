import json
import modules
import time
import torch as th
import torchvision as tv
from torch.utils.data import DataLoader
import utils


@th.inference_mode()
def main(batch_size: int,
         config_path: str,
         save_path: str,
         data_path: str,
         num_workers: int) -> None:

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = tv.datasets.ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            pin_memory=True,
                            num_workers=num_workers)
    print(f"{len(dataset)} images in {data_path}, {len(dataloader)} batches of size {batch_size}.")

    with open(config_path, "r") as f:
        config = json.load(f)
    model = modules.CascadedWaveletDiffuser(num_channels=config["num_channels"],
                                            wd_cfgs=config["levels"]).to(device)
    ema = th.optim.swa_utils.AveragedModel(model,
                                           multi_avg_fn=th.optim.swa_utils.get_ema_multi_avg_fn(
                                                config["ema_decay"]))
    optim = th.optim.Adam(model.parameters(), lr=config["learning_rate"])
    print(f"Loaded {config_path} model with {utils.count_parameters(model)} parameters.")

    start = time.perf_counter()
    model.calculate_stats(dataloader)
    utils.save_state(checkpoint_path=save_path,
                     model=model,
                     ema=ema,
                     optimizer=optim)
    print(f"Calculated stats in {time.perf_counter() - start:.2f} seconds.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers")
    parser.add_argument("--config_path", type=str, required=True, help="path to config")
    parser.add_argument("--save_path", type=str, required=True, help="path to save")
    parser.add_argument("--data_path", type=str, required=True, help="path to data")
    args = parser.parse_args()

    world_size = th.cuda.device_count()
    main(batch_size=args.batch_size,
         config_path=args.config_path,
         save_path=args.save_path,
         data_path=args.data_path,
         num_workers=args.num_workers)
