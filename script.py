#!/usr/bin/env python3
"""
Prokudin-Gorskii channel alignment (SYDE 671, Assignment 1, Part 2).

Splits a stacked B/G/R glass-plate scan into three channels, aligns G and R
onto B, and saves the color composite. Uses one search function for both
modes: single-scale calls it once; the pyramid calls it recursively.

Usage
-----
    python script.py plate.tif                       # pyramid, L2 metric
    python script.py plate.tif --metric ncc          # pyramid, NCC metric
    python script.py plate.jpg --single-scale        # exhaustive, small images only
    python script.py plate.tif --output out/emir.jpg
"""

import argparse
import os

import numpy as np
from PIL import Image


# Setup

def load_gray(path):
    """Load an image as a float32 grayscale array in [0, 1] (handles uint8/uint16)."""
    arr = np.array(Image.open(path))
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return (arr / scale).astype(np.float32)


def split_channels(img):
    """Split a vertically stacked plate into equal-height Blue, Green, Red (top to bottom)."""
    h = img.shape[0] // 3
    return img[:h], img[h:2 * h], img[2 * h:3 * h]


def normalize(img):
    """Zero-mean, unit-std, so channels with different brightness are comparable."""
    return (img - img.mean()) / (img.std() + 1e-8)


def interior(img, margin=0.1):
    """Drop `margin` of each side so plate borders / roll wraparound don't affect scoring."""
    h, w = img.shape
    my, mx = int(h * margin), int(w * margin)
    return img[my:h - my, mx:w - mx]


def downsample(img):
    """Halve resolution by averaging 2x2 blocks."""
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    return img[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


# Alignment

def score(a, b, metric):
    """Similarity of two equal-size arrays. Higher score is better."""
    if metric == "l2":
        return -np.mean((a - b) ** 2)
    return np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)  # ncc


def search(ref, mov, center, radius, metric):
    """Try every (dy, dx) within radius of center; return the best-scoring shift."""
    ref = normalize(ref)
    mov = normalize(mov)
    ref_in = interior(ref)

    best, best_shift = -np.inf, center
    for dy in range(center[0] - radius, center[0] + radius + 1):
        for dx in range(center[1] - radius, center[1] + radius + 1):
            shifted = np.roll(mov, (dy, dx), axis=(0, 1))
            s = score(ref_in, interior(shifted), metric)
            if s > best:
                best, best_shift = s, (dy, dx)
    return best_shift


def pyramid_align(ref, mov, metric, radius=15, refine=2, min_size=300):
    """Solve on a half-size image, double, refine."""
    if min(ref.shape) < min_size:
        shift = search(ref, mov, (0, 0), radius, metric)   # coarsest level: wide search
    else:
        coarse = pyramid_align(downsample(ref), downsample(mov),
                               metric, radius, refine, min_size)
        shift = search(ref, mov, (2 * coarse[0], 2 * coarse[1]), refine, metric)
    print(f"  {ref.shape[1]}x{ref.shape[0]}: shift (dy, dx) = {shift}")
    return shift


def crop_borders(rgb, rel=0.5, max_frac=0.12, pad=0.005):
    """Trim the dark plate frame and the odd-colored strips next to it."""
    def cut(line):                       # line: (n, 3) per-channel means
        n = len(line)
        k = int(n * max_frac)
        typical = np.median(line[n // 4: 3 * n // 4], axis=0)
        bad = (line < rel * typical).any(axis=1)
        head = np.where(bad[:k])[0]
        tail = np.where(bad[n - k:])[0]
        p = int(n * pad)
        start = head[-1] + 1 + p if len(head) else 0
        stop = n - k + tail[0] - p if len(tail) else n
        return start, stop

    y0, y1 = cut(rgb.mean(axis=1))
    x0, x1 = cut(rgb.mean(axis=0))
    return rgb[y0:y1, x0:x1]



# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the stacked B/G/R plate image")
    parser.add_argument("--output", default="aligned.jpg", help="Output color image (default: aligned.jpg)")
    parser.add_argument("--metric", choices=["l2", "ncc"], default="l2")
    parser.add_argument("--single-scale", action="store_true",
                        help="One exhaustive search of +/- --radius px (small images only)")
    parser.add_argument("--radius", type=int, default=15, help="Search radius in pixels (default: 15)")
    args = parser.parse_args()

    B, G, R = split_channels(load_gray(args.input))
    print(f"Split into B/G/R channels, each {B.shape[1]}x{B.shape[0]}")

    shifts = {}
    for name, ch in (("G", G), ("R", R)):
        print(f"\nAligning {name} -> B (metric={args.metric})")
        if args.single_scale:
            shifts[name] = search(B, ch, (0, 0), args.radius, args.metric)
        else:
            shifts[name] = pyramid_align(B, ch, args.metric, radius=args.radius)

    print()
    for name, (dy, dx) in shifts.items():
        print(f"{name} displacement (x, y) = ({dx}, {dy})")

    G_al = np.roll(G, shifts["G"], axis=(0, 1))
    R_al = np.roll(R, shifts["R"], axis=(0, 1))

    # trim the wrapped-around edges left by np.roll
    h, w = B.shape
    cy = max(abs(shifts["G"][0]), abs(shifts["R"][0]))
    cx = max(abs(shifts["G"][1]), abs(shifts["R"][1]))
    rgb = np.dstack([R_al, G_al, B])[cy:h - cy, cx:w - cx]
    rgb = crop_borders(rgb)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(args.output)

    # also save the split channels for the webpage
    stem = os.path.splitext(args.output)[0]
    for name, ch in (("B", B), ("G", G), ("R", R)):
        Image.fromarray((ch * 255).astype(np.uint8)).save(f"{stem}_{name}.jpg")

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()