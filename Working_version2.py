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
    python script.py plate.tif --no-crop             # keep plate borders
    python script.py plate.tif --show-pyramid        # save an overlay, diff, and 4-panel figure per pyramid level
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------
# I/O and setup
# --------------------------------------------------

def load_gray(path):
    """Load an image as a float32 grayscale array in [0, 1] (handles uint8/uint16)."""
    arr = np.array(Image.open(path))
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return (arr / scale).astype(np.float32)


def split_channels(img):
    """Split a vertically stacked plate into equal-height B, G, R (top to bottom)."""
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


# --------------------------------------------------
# Alignment
# --------------------------------------------------

def score(a, b, metric):
    """Similarity of two equal-size arrays. Higher is always better."""
    if metric == "l2":
        return -np.mean((a - b) ** 2)
    return np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)  # ncc


def search(ref, mov, center, radius, metric):
    """Try every (dy, dx) within `radius` of `center`; return the best-scoring shift."""
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


def overlay(ref, mov, shift):
    """Red/cyan overlay of ref against shifted mov, for visualizing alignment quality."""
    aligned = np.roll(mov, shift, axis=(0, 1))
    rgb = np.dstack([aligned, ref, ref])   # mov -> red, ref -> cyan; aligned regions look gray
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def diff_image(ref, mov, shift, gain=3.0):
    """Grayscale |ref - shifted mov|, brightened by `gain` so small misalignments are visible.

    A well-aligned pair is mostly black; bright edges mark where the two still
    disagree (misalignment, or genuine per-channel brightness differences).
    """
    aligned = np.roll(mov, shift, axis=(0, 1))
    d = np.clip(np.abs(ref - aligned) * gain, 0, 1)
    return (d * 255).astype(np.uint8)


def panel_figure(ref, mov, shift, panel_width=280, diff_gain=3.0):
    """Four-panel comparison, resized to a fixed display width: reference, moving
    (before alignment), moving (after alignment), and their brightened |difference|.
    """
    aligned = np.roll(mov, shift, axis=(0, 1))
    diff = np.clip(np.abs(ref - aligned) * diff_gain, 0, 1)
    panels = [ref, mov, aligned, diff]
    titles = ["Reference (B)", "Moving, before", "Moving, aligned", f"|difference| x{diff_gain:g}"]

    h, w = ref.shape
    ph = max(1, round(h * panel_width / w))
    gap, bar = 6, 20
    font = ImageFont.load_default()

    canvas = Image.new("RGB", (panel_width * 4 + gap * 3, ph + bar), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for arr, title in zip(panels, titles):
        tile = Image.fromarray((arr * 255).astype(np.uint8)).resize((panel_width, ph), Image.LANCZOS)
        canvas.paste(tile, (x, bar))
        draw.text((x + 4, 4), title, fill=(255, 255, 255), font=font)
        x += panel_width + gap
    return canvas


def save_diagnostics(ref, mov, shift, prefix):
    """Save the overlay, difference, and 4-panel comparison for one alignment result."""
    h, w = ref.shape
    Image.fromarray(overlay(ref, mov, shift)).save(f"{prefix}_overlay_{h}x{w}.jpg")
    Image.fromarray(diff_image(ref, mov, shift)).save(f"{prefix}_diff_{h}x{w}.jpg")
    panel_figure(ref, mov, shift).save(f"{prefix}_panels_{h}x{w}.jpg")


def pyramid_align(ref, mov, metric, radius=15, refine=2, min_size=300, save_prefix=None, _counter=None):
    """Coarse-to-fine alignment: solve on a half-size image, double, refine by +/-`refine`.

    If `save_prefix` is given, saves diagnostics (see `save_diagnostics`) for every
    level, numbered L0 (coarsest) to Lmax (full resolution) -- useful for showing
    how the estimate improves down the pyramid.
    """
    top_call = _counter is None
    if top_call:
        _counter = [0]   # shared across the recursion; counts up coarse -> fine

    if min(ref.shape) < min_size:
        shift = search(ref, mov, (0, 0), radius, metric)   # coarsest level: wide search
    else:
        coarse = pyramid_align(downsample(ref), downsample(mov), metric, radius, refine,
                               min_size, save_prefix, _counter)
        shift = search(ref, mov, (2 * coarse[0], 2 * coarse[1]), refine, metric)

    aligned = np.roll(mov, shift, axis=(0, 1))
    l2 = np.mean((interior(ref) - interior(aligned)) ** 2)
    ncc = score(normalize(interior(ref)), normalize(interior(aligned)), "ncc")
    print(f"  {ref.shape[1]}x{ref.shape[0]}: shift (dy, dx) = {shift}, L2 = {l2:.5f}, NCC = {ncc:.4f}")

    if save_prefix is not None:
        save_diagnostics(ref, mov, shift, f"{save_prefix}_L{_counter[0]}")
        _counter[0] += 1
    return shift


def crop_borders(rgb, k=2.0, max_frac=0.12, pad=0.005):
    """Detect and trim the plate border using disagreement between color channels.

    Inside the picture the three aligned channels show the same structure; at
    the plate frame they don't (the black frame sits at a different place in
    each exposure), which is what produces the colored strips. Each channel is
    normalized, then per-pixel disagreement |R-B| + |G-B| is averaged along every
    row and column. A line is "border" if its disagreement exceeds `k` x the
    median over the central half of the image. Only the outer `max_frac` of each
    side is searched; we cut at the innermost border line found there, plus a
    small `pad` for the soft edge.
    """
    n = (rgb - rgb.mean(axis=(0, 1))) / (rgb.std(axis=(0, 1)) + 1e-8)
    d = np.abs(n[..., 0] - n[..., 2]) + np.abs(n[..., 1] - n[..., 2])

    def cut(line):
        m = len(line)
        edge = int(m * max_frac)
        bad = line > k * np.median(line[m // 4: 3 * m // 4])
        head = np.where(bad[:edge])[0]
        tail = np.where(bad[m - edge:])[0]
        p = int(m * pad)
        start = head[-1] + 1 + p if len(head) else 0
        stop = m - edge + tail[0] - p if len(tail) else m
        return start, stop

    y0, y1 = cut(d.mean(axis=1))
    x0, x1 = cut(d.mean(axis=0))
    return rgb[y0:y1, x0:x1]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the stacked B/G/R plate image")
    parser.add_argument("--output", default="aligned.jpg", help="Output color image (default: aligned.jpg)")
    parser.add_argument("--metric", choices=["l2", "ncc"], default="l2")
    parser.add_argument("--single-scale", action="store_true",
                        help="One exhaustive search of +/- --radius px (small images only)")
    parser.add_argument("--radius", type=int, default=15, help="Search radius in pixels (default: 15)")
    parser.add_argument("--no-crop", action="store_true",
                        help="Keep the plate borders (skip automatic border cropping)")
    parser.add_argument("--show-pyramid", action="store_true",
                        help="Save an overlay, difference image, and 4-panel comparison figure, and "
                             "print the L2/NCC score, at each pyramid level (a single level under --single-scale)")
    args = parser.parse_args()

    B, G, R = split_channels(load_gray(args.input))
    print(f"Split into B/G/R channels, each {B.shape[1]}x{B.shape[0]}")

    stem = os.path.splitext(args.output)[0]

    shifts = {}
    for name, ch in (("G", G), ("R", R)):
        print(f"\nAligning {name} -> B (metric={args.metric})")
        if args.single_scale:
            shift = search(B, ch, (0, 0), args.radius, args.metric)
            aligned = np.roll(ch, shift, axis=(0, 1))
            l2 = np.mean((interior(B) - interior(aligned)) ** 2)
            ncc = score(normalize(interior(B)), normalize(interior(aligned)), "ncc")
            print(f"  {B.shape[1]}x{B.shape[0]}: shift (dy, dx) = {shift}, L2 = {l2:.5f}, NCC = {ncc:.4f}")
            if args.show_pyramid:
                save_diagnostics(B, ch, shift, f"{stem}_pyramid_{name}_L0")
            shifts[name] = shift
        else:
            prefix = f"{stem}_pyramid_{name}" if args.show_pyramid else None
            shifts[name] = pyramid_align(B, ch, args.metric, radius=args.radius, save_prefix=prefix)

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

    if not args.no_crop:
        rgb = crop_borders(rgb)   # detect and trim the plate border

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(args.output)

    # also save the split channels for the webpage
    for name, ch in (("B", B), ("G", G), ("R", R)):
        Image.fromarray((ch * 255).astype(np.uint8)).save(f"{stem}_{name}.jpg")

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()