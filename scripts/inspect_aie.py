import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect top-k attention heads by AIE.")
    parser.add_argument("--aie", required=True, help="Path to an AIE .pt file.")
    parser.add_argument("--topk", type=int, default=10, help="Number of heads to print.")
    args = parser.parse_args()
    if args.topk <= 0:
        parser.error("--topk must be positive")
    return args


def main():
    args = parse_args()
    aie = torch.load(args.aie, map_location="cpu")
    if not torch.is_tensor(aie):
        raise TypeError(f"Expected a tensor in {args.aie}, got {type(aie).__name__}")
    if aie.ndim == 3 and aie.shape[-1] == 1:
        aie = aie.squeeze(-1)
    if aie.ndim != 2:
        raise ValueError(f"Expected AIE tensor with shape [layers, heads], got {tuple(aie.shape)}")

    flat = aie.flatten()
    topk = min(args.topk, flat.numel())
    values, indices = torch.topk(flat, k=topk)
    n_heads = aie.shape[1]

    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        layer = index // n_heads
        head = index % n_heads
        print(f"rank={rank} layer={layer} head={head} AIE={value:.8g}")


if __name__ == "__main__":
    main()
