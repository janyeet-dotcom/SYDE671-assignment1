#!/usr/bin/env python3
"""
Prokudin-Gorskii channel alignment (SYDE 671, Assignment 1, Part 2).

Splits a glass-plate scan (three stacked B/G/R exposures, top to bottom)
into its three channels, aligns G and R onto B using either a single-scale
exhaustive search or a coarse-to-fine image pyramid, and saves the
composite color image. Supports both the L2 and NCC matching metrics
required by the assignment.

Usage
-----
    # pyramid alignment (recommended for full-size images), L2 metric
    python script.py plate.jpg --output aligned.png

    # NCC metric instead of L2
    python script.py plate.jpg --output aligned.png --metric ncc

    # single-scale exhaustive search (only practical on small/low-res images)
    python script.py plate.jpg --output aligned.png --single-scale --radius 15

    # watch each pyramid level as it aligns
    python script.py plate.jpg --output aligned.png --show-pyramid

    # also save the split B/G/R channel images (handy for a report/webpage)
    python script.py plate.jpg --output aligned.png --save-channels
"""

import argparse
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# --------------------------------------------------
# I/O
# --------------------------------------------------

def load_grayscale(path):
    """Load an image and convert it to 8-bit grayscale."""
    return np.array(Image.open(path).convert("L"))


def save_rgb(rgb, path):
    """Save an (H, W, 3) array as a color image, creating the output
    directory first if it doesn't exist yet.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


# --------------------------------------------------
# Split stacked plate
# --------------------------------------------------

def split_channels(img):
    """Split a vertically stacked B/G/R plate (top to bottom, per the
    assignment) into its three equal-height channel images.
    """
    h = img.shape[0] // 3

    B = img[0:h]
    G = img[h:2 * h]
    R = img[2 * h:3 * h]

    min_h = min(B.shape[0], G.shape[0], R.shape[0])

    return (
        B[:min_h],
        G[:min_h],
        R[:min_h]
    )


# --------------------------------------------------
# Image utilities
# --------------------------------------------------

def resize_image(img, scale):
    """Resize `img` by `scale` (Lanczos resampling)."""
    h, w = img.shape

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    return np.array(
        Image.fromarray(img).resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS
        )
    )


def shift_image(img, dy, dx):
    """Translate `img` by (dy, dx), filling exposed edges with 0 (black)."""
    h, w = img.shape

    out = np.zeros_like(img)

    src_y0 = max(0, -dy)
    src_y1 = h - max(0, dy)

    src_x0 = max(0, -dx)
    src_x1 = w - max(0, dx)

    dst_y0 = max(0, dy)
    dst_y1 = h - max(0, -dy)

    dst_x0 = max(0, dx)
    dst_x1 = w - max(0, -dx)

    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out

    out[
        dst_y0:dst_y1,
        dst_x0:dst_x1
    ] = img[
        src_y0:src_y1,
        src_x0:src_x1
    ]

    return out


def crop_common_region(shifts, shape):
    """Given the (dy, dx) shift applied to each channel (use (0, 0) for
    whichever channel is the fixed reference), return the crop bounds
    (y0, y1, x0, x1) that exclude every channel's black shifted-in edges,
    keeping only the region all three channels actually cover.
    """
    h, w = shape
    dys = [s[0] for s in shifts]
    dxs = [s[1] for s in shifts]

    y0 = max([0] + dys)
    y1 = h + min([0] + dys)
    x0 = max([0] + dxs)
    x1 = w + min([0] + dxs)

    y0, y1 = max(0, y0), max(y0, y1)
    x0, x1 = max(0, x0), max(x0, x1)
    return y0, y1, x0, x1


def normalize(img):
    """Zero-mean, unit-std version of `img`, as float64.

    The B/G/R channels of a Prokudin-Gorskii plate routinely have quite
    different overall brightness and contrast (the filters weren't equally
    sensitive), which biases a plain intensity-difference metric. Comparing
    normalized images instead keeps the score driven by actual structural
    (mis)match rather than by which shift happens to average out the
    brightness gap.
    """
    img = img.astype(np.float64)

    std = img.std()

    if std < 1e-8:
        return img - img.mean()

    return (img - img.mean()) / std


# --------------------------------------------------
# Interior crop for scoring
# --------------------------------------------------

def center_region(img, keep_fraction=0.8):
    """Crop out the center `keep_fraction` of `img`, trimming the rest as a
    margin. Used to keep border artifacts from influencing the alignment
    score (the assignment recommends scoring only on interior pixels).
    """
    h, w = img.shape

    margin_y = int((1.0 - keep_fraction) * h / 2.0)
    margin_x = int((1.0 - keep_fraction) * w / 2.0)

    return img[
        margin_y:h - margin_y,
        margin_x:w - margin_x
    ]


# --------------------------------------------------
# Overlap extraction
# --------------------------------------------------

def overlapping_regions(ref, mov, dy, dx):
    """Return the (interior-cropped) overlapping regions of `ref` and `mov`
    shifted by (dy, dx), or (None, None) if they don't overlap at all.
    """
    h, w = ref.shape

    ref_y0 = max(0, dy)
    ref_y1 = min(h, h + dy)

    ref_x0 = max(0, dx)
    ref_x1 = min(w, w + dx)

    mov_y0 = max(0, -dy)
    mov_y1 = min(h, h - dy)

    mov_x0 = max(0, -dx)
    mov_x1 = min(w, w - dx)

    if ref_y1 <= ref_y0 or ref_x1 <= ref_x0:
        return None, None

    r = ref[
        ref_y0:ref_y1,
        ref_x0:ref_x1
    ]

    m = mov[
        mov_y0:mov_y1,
        mov_x0:mov_x1
    ]

    r = center_region(r)
    m = center_region(m)

    return r, m


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def l2_score(ref, mov, dy, dx):
    """Mean squared difference over the overlap. Lower is better."""
    r, m = overlapping_regions(ref, mov, dy, dx)

    if r is None:
        return np.inf

    return np.mean((r - m) ** 2)


def ncc_score(ref, mov, dy, dx):
    """Normalized cross-correlation over the overlap. Higher is better."""
    r, m = overlapping_regions(ref, mov, dy, dx)

    if r is None:
        return -np.inf

    r = r.ravel()
    m = m.ravel()

    denom = np.linalg.norm(r) * np.linalg.norm(m)

    if denom < 1e-12:
        return -np.inf

    return np.dot(r, m) / denom


# --------------------------------------------------
# Visualization
# --------------------------------------------------

def show_stage(ref, mov, dy, dx, factor, score, title):
    """Show reference / moving / shifted / overlay panels for one pyramid
    level. `ref` and `mov` must be raw (non-normalized) images -- pass the
    resized-but-unnormalized versions, not the ones used for scoring, or
    the overlay panel below will render as noise (normalized data is
    zero-mean and can go negative, which wraps around when forced into a
    uint8 image instead of clipping sensibly).
    """
    shifted = shift_image(mov, dy, dx)

    overlay = np.zeros(
        (ref.shape[0], ref.shape[1], 3),
        dtype=np.uint8
    )

    overlay[..., 0] = ref
    overlay[..., 1] = shifted

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    ax[0, 0].imshow(ref, cmap="gray")
    ax[0, 0].set_title("Reference")

    ax[0, 1].imshow(mov, cmap="gray")
    ax[0, 1].set_title("Moving")

    ax[1, 0].imshow(shifted, cmap="gray")
    ax[1, 0].set_title(f"Shifted ({dy}, {dx})")

    ax[1, 1].imshow(overlay)
    ax[1, 1].set_title("Overlay")

    for a in ax.ravel():
        a.axis("off")

    fig.suptitle(
        f"{title}\n"
        f"Scale=1/{factor} "
        f"Shift=({dy},{dx}) "
        f"Score={score:.6f}"
    )

    plt.tight_layout()

    # show but DO NOT block
    plt.show(block=False)

    # bring figure to front
    plt.pause(0.1)

    try:
        input("\nPress ENTER for next pyramid level...")
    except EOFError:
        # no interactive stdin available (e.g. running non-interactively) --
        # don't hang, just move on
        pass

    plt.close(fig)


# --------------------------------------------------
# Single-scale exhaustive alignment
# --------------------------------------------------

def exhaustive_align(
        ref,
        mov,
        radius,
        metric="l2"):
    """Exhaustively search every (dy, dx) in [-radius, radius] and return
    the shift (and its score) that best aligns `mov` onto `ref`.
    """
    ref = normalize(ref)
    mov = normalize(mov)

    best_shift = (0, 0)

    if metric == "l2":
        best_score = np.inf
    else:
        best_score = -np.inf

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):

            if metric == "l2":
                score = l2_score(ref, mov, dy, dx)

                if score < best_score:
                    best_score = score
                    best_shift = (dy, dx)

            else:
                score = ncc_score(ref, mov, dy, dx)

                if score > best_score:
                    best_score = score
                    best_shift = (dy, dx)

    return best_shift, best_score


# --------------------------------------------------
# Pyramid alignment
# --------------------------------------------------

def pyramid_align(
        ref,
        mov,
        metric="l2",
        search_radius=4,
        visualize=False,
        max_shift_frac=0.10):
    """Coarse-to-fine image-pyramid alignment.

    At each scale (coarsest to finest), downscale both images, normalize
    them, search a window of candidate shifts around the running estimate,
    and keep whichever minimizes L2 / maximizes NCC. The estimate is
    doubled going into each finer level.

    `max_shift_frac` bounds the TOTAL shift (at full resolution, in either
    axis) that the search is ever allowed to settle on, at every level --
    not just the first. Without this, a bad guess at the coarse level has
    nothing stopping it from being doubled into an arbitrarily large, wrong
    shift by the time it reaches full resolution (this is the classic
    coarse-to-fine "runaway" failure, and it's much more likely on
    low-texture scenes -- wide skies, flat water -- than on
    higher-contrast portraits, since there's less structure at the coarse
    level to pin down the right answer). A true Prokudin-Gorskii plate
    misalignment is always a small fraction of the frame, so this is a safe
    assumption, not just a band-aid.
    """
    levels = [64, 32, 16, 8, 4, 2, 1]

    dy = 0
    dx = 0

    short_side = min(ref.shape)

    levels = [
        l for l in levels
        if l == 1 or short_side / l >= 8
    ]

    h, w = ref.shape
    max_dy = max(1, int(round(max_shift_frac * h)))
    max_dx = max(1, int(round(max_shift_frac * w)))

    for idx, factor in enumerate(levels):

        scale = 1.0 / factor

        # keep a raw (unnormalized) copy for the visualization -- scoring
        # uses the normalized version below
        ref_small_raw = resize_image(ref, scale)
        mov_small_raw = resize_image(mov, scale)

        ref_small = normalize(ref_small_raw)
        mov_small = normalize(mov_small_raw)

        level_max_dy = max(1, int(round(max_dy / factor)))
        level_max_dx = max(1, int(round(max_dx / factor)))

        if idx == 0:
            base_radius = max(
                search_radius,
                int(min(ref_small.shape) * 0.25)
            )
        else:
            base_radius = search_radius

        radius_y = min(base_radius, level_max_dy)
        radius_x = min(base_radius, level_max_dx)

        best_shift = (dy, dx)

        if metric == "l2":
            best_score = np.inf
        else:
            best_score = -np.inf

        for ddy in range(-radius_y, radius_y + 1):
            for ddx in range(-radius_x, radius_x + 1):

                test_dy = int(np.clip(dy + ddy, -level_max_dy, level_max_dy))
                test_dx = int(np.clip(dx + ddx, -level_max_dx, level_max_dx))

                if metric == "l2":
                    score = l2_score(
                        ref_small,
                        mov_small,
                        test_dy,
                        test_dx
                    )

                    if score < best_score:
                        best_score = score
                        best_shift = (
                            test_dy,
                            test_dx
                        )
                else:
                    score = ncc_score(
                        ref_small,
                        mov_small,
                        test_dy,
                        test_dx
                    )

                    if score > best_score:
                        best_score = score
                        best_shift = (
                            test_dy,
                            test_dx
                        )

        dy, dx = best_shift

        print(
            f"  Scale 1/{factor:<3} "
            f"Shift=({dy},{dx}) "
            f"Score={best_score:.6f}"
        )

        if visualize:
            show_stage(
                ref_small_raw,
                mov_small_raw,
                dy,
                dx,
                factor,
                best_score,
                "Pyramid Alignment"
            )

        if factor > 1:
            dy *= 2
            dx *= 2
            # keep the doubled estimate within bounds for the next (finer)
            # level too
            dy = int(np.clip(dy, -max_dy, max_dy))
            dx = int(np.clip(dx, -max_dx, max_dx))

    return dy, dx


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", help="Path to the input B/G/R plate image")

    parser.add_argument(
        "--output",
        default="aligned.png",
        help="Path to save the aligned color composite (default: aligned.png)"
    )

    parser.add_argument(
        "--metric",
        choices=["l2", "ncc"],
        default="l2",
        help="Matching metric: l2 (sum/mean squared difference, lower is "
             "better) or ncc (normalized cross-correlation, higher is "
             "better). Default: l2"
    )

    parser.add_argument(
        "--single-scale",
        action="store_true",
        help="Use a single exhaustive search over +/- --radius pixels "
             "instead of the coarse-to-fine pyramid. Only practical on "
             "small/low-res images -- cost grows with radius squared."
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=15,
        help="Search radius in pixels for --single-scale mode (default: 15)"
    )

    parser.add_argument(
        "--pyramid-radius",
        type=int,
        default=4,
        help="Search radius (pixels, at each level's own resolution) used "
             "to refine the estimate at every pyramid level after the "
             "first (default: 4)"
    )

    parser.add_argument(
        "--max-shift-frac",
        type=float,
        default=0.10,
        help="Pyramid mode only: cap the total shift, at full resolution, "
             "to this fraction of the image in either axis, at every "
             "level -- prevents a bad coarse-level guess from diverging "
             "into a much larger wrong shift (default: 0.10)"
    )

    parser.add_argument(
        "--show-pyramid",
        action="store_true",
        help="Display each pyramid level's alignment as it happens "
             "(pyramid mode only); press Enter to advance"
    )

    parser.add_argument(
        "--save-channels",
        action="store_true",
        help="Also save the split B/G/R channel images (imageB.png, "
             "imageG.png, imageR.png) next to --output"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        raise SystemExit(1)

    print(f"Loading {args.input}...")
    img = load_grayscale(args.input)

    B, G, R = split_channels(img)
    print(f"Split into B/G/R channels, each {B.shape[1]}x{B.shape[0]}")

    if args.save_channels:
        out_dir = os.path.dirname(args.output) or "."
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(B).save(os.path.join(out_dir, "imageB.png"))
        Image.fromarray(G).save(os.path.join(out_dir, "imageG.png"))
        Image.fromarray(R).save(os.path.join(out_dir, "imageR.png"))
        print(f"Saved channel images to {out_dir}/")

    print(f"\nAligning G -> B (metric={args.metric})")
    if args.single_scale:
        g_shift, _ = exhaustive_align(B, G, args.radius, args.metric)
    else:
        g_shift = pyramid_align(
            B, G,
            metric=args.metric,
            search_radius=args.pyramid_radius,
            visualize=args.show_pyramid,
            max_shift_frac=args.max_shift_frac,
        )

    print(f"\nAligning R -> B (metric={args.metric})")
    if args.single_scale:
        r_shift, _ = exhaustive_align(B, R, args.radius, args.metric)
    else:
        r_shift = pyramid_align(
            B, R,
            metric=args.metric,
            search_radius=args.pyramid_radius,
            visualize=args.show_pyramid,
            max_shift_frac=args.max_shift_frac,
        )

    # assignment asks for the displacement printed as (x, y) -- note this is
    # the reverse of the (dy, dx) = (row, col) order used internally
    print()
    print(f"G displacement (x, y) = ({g_shift[1]}, {g_shift[0]})")
    print(f"R displacement (x, y) = ({r_shift[1]}, {r_shift[0]})")

    G_aligned = shift_image(G, *g_shift)
    R_aligned = shift_image(R, *r_shift)

    # crop away the black edges that shifting exposes, so only the region
    # all three channels actually cover survives into the final composite
    y0, y1, x0, x1 = crop_common_region([(0, 0), g_shift, r_shift], B.shape)
    if y1 <= y0 or x1 <= x0:
        print("Warning: shifts leave no common region; skipping crop.")
        y0, y1, x0, x1 = 0, B.shape[0], 0, B.shape[1]

    rgb = np.dstack([
        R_aligned[y0:y1, x0:x1],
        G_aligned[y0:y1, x0:x1],
        B[y0:y1, x0:x1],
    ])

    save_rgb(rgb, args.output)

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()