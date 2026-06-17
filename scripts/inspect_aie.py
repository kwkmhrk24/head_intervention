import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description="Inspect top attention heads by AIE score.")
    parser.add_argument("--aie", required=True, help="Path to an AIE .pt file.")
    parser.add_argument("--topk", type=int, default=10, help="Number of top heads to print.")
    return parser.parse_args()


def load_aie(path):
    aie = torch.load(path, map_location="cpu")
    if not torch.is_tensor(aie):
        raise TypeError(f"Expected a torch.Tensor, got {type(aie).__name__}")

    if aie.ndim == 3 and aie.shape[-1] == 1:
        aie = aie.squeeze(-1)

    if aie.ndim != 2:
        raise ValueError(f"Expected AIE shape [layers, heads], got {tuple(aie.shape)}")

    return aie


def main():
    args = get_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive")

    aie = load_aie(args.aie)
    layers, heads = aie.shape
    topk = min(args.topk, aie.numel())
    values, flat_indices = torch.topk(aie.reshape(-1), k=topk)

    for rank, (value, flat_index) in enumerate(zip(values, flat_indices), start=1):
        layer = int(flat_index.item() // heads)
        head = int(flat_index.item() % heads)
        print(f"rank={rank} layer={layer} head={head} AIE={value.item():.8g}")


if __name__ == "__main__":
    main()
