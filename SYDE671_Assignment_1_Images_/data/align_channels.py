#!/usr/bin/env python3
"""
align_channels.py
==================

Splits a Prokudin-Gorskii-style photograph (a single image containing three
stacked exposures of the same scene, taken through red, green, and blue
filters) into its three channel images, lets the user click matching
"anchor points" in each channel, and then aligns the three channels using
a local, anchor-seeded cross-correlation / difference-minimization search.
The final composite is cropped so that only the region covered by all
three channels remains.

Pipeline
--------
1.  Load the input image and convert it to 8-bit grayscale.
2.  Remove the black film/scan edges around the whole strip.
3.  Split the strip into three roughly-equal images (R, G, B) and save them
    as imageR.png, imageG.png, imageB.png.
4.  Clean up any leftover thin border artifacts on each individual channel.
5.  Display the three channel images side-by-side with pixel-coordinate
    scale bars (axes ticks), so the user can identify good anchor points.
6.  Let the user click one (or more) corresponding anchor point(s) in each
    of the three channel images.
7.  Use the anchor points to compute an initial (dy, dx) guess for how far
    G and B must be shifted to line up with R, then refine that guess with
    a local search that minimizes a sum-of-squared-differences function
    (i.e. maximizes the cross-correlation) in a window around the guess.
8.  Shift G and B onto R, build the RGB composite, and crop away any
    border region that isn't covered by all three shifted channels.

Usage
-----
    python align_channels.py input.jpg --output-dir out/

    python align_channels.py input.jpg --output-dir out/ \
        --n-anchors 2 --search-range 20 --no-display

    # If automatic border detection isn't cropping cleanly, either loosen/
    # tighten its thresholds:
    python align_channels.py input.jpg --output-dir out/ \
        --split-dark-frac 0.8 --residual-threshold 20

    # ...or bypass it entirely and drag-select the crop by hand:
    python align_channels.py input.jpg --output-dir out/ --manual-crop

Dependencies: numpy, pillow, matplotlib, scipy
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


# --------------------------------------------------------------------------
# Manual crop: drag-select a rectangle instead of relying on automatic
# black-border detection. Useful when a scan's border isn't cleanly black,
# is textured, or the automatic threshold is cropping too much/too little.
# --------------------------------------------------------------------------

def manual_crop(img, title):
    """Display `img` and let the user drag a rectangle marking the region to
    KEEP. Returns the cropped array, or `img` unchanged if the user closes
    the window without dragging a rectangle.
    """
    selection = {}

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap="gray")
    ax.set_title(title + "\n(drag a box around the region to KEEP, then close this window)")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    def onselect(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        selection["rect"] = (min(y0, y1), max(y0, y1), min(x0, x1), max(x0, x1))

    # keep a reference alive so it isn't garbage-collected before use
    _selector = RectangleSelector(
        ax, onselect, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords="pixels", interactive=True,
    )
    plt.show()

    if "rect" not in selection:
        print(f"  (no rectangle drawn for '{title}' -- keeping image unchanged)")
        return img

    h, w = img.shape
    y0, y1, x0, x1 = selection["rect"]
    y0, y1 = max(0, int(round(y0))), min(h, int(round(y1)))
    x0, x1 = max(0, int(round(x0))), min(w, int(round(x1)))
    if y1 <= y0 or x1 <= x0:
        return img
    return img[y0:y1, x0:x1]


def manual_crop_shared(images, title):
    """Like `manual_crop`, but the rectangle is drawn once (on the first
    image) and applied identically to every image in `images`, keeping
    their coordinate frames in sync.
    """
    selection = {}
    ref = images[0]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(ref, cmap="gray")
    ax.set_title(title + "\n(drag a box around the region to KEEP -- applies to all channels, then close)")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    def onselect(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        selection["rect"] = (min(y0, y1), max(y0, y1), min(x0, x1), max(x0, x1))

    _selector = RectangleSelector(
        ax, onselect, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords="pixels", interactive=True,
    )
    plt.show()

    if "rect" not in selection:
        print(f"  (no rectangle drawn for '{title}' -- keeping images unchanged)")
        return list(images)

    h, w = ref.shape
    y0, y1, x0, x1 = selection["rect"]
    y0, y1 = max(0, int(round(y0))), min(h, int(round(y1)))
    x0, x1 = max(0, int(round(x0))), min(w, int(round(x1)))
    if y1 <= y0 or x1 <= x0:
        return list(images)
    return [img[y0:y1, x0:x1] for img in images]


# --------------------------------------------------------------------------
# Step 1-2: load + black-edge removal
# --------------------------------------------------------------------------

def load_grayscale(path):
    """Load an image and convert it to 8-bit grayscale."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def remove_black_edges(img, threshold=10):
    """Crop rows/columns from the border that are essentially all black.

    Scans in from each of the four sides and stops at the first row/column
    whose max value exceeds `threshold`.
    """
    mask = img > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img  # nothing but black; return unchanged
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin:rmax + 1, cmin:cmax + 1]


# --------------------------------------------------------------------------
# Step 3: split the vertical strip into R, G, B
# --------------------------------------------------------------------------

def split_vertical_strip(img, dark_threshold=10, dark_frac_cutoff=0.9):
    """Split a vertical strip into its three channel sub-images.

    Historic Prokudin-Gorskii negatives are ordered (top to bottom) as
    Blue, Green, Red; this returns them in that top/middle/bottom order
    and the caller decides which physical channel each corresponds to.

    Rather than assuming the strip splits into exact equal thirds, this
    finds the three actual content (non-border) bands by flagging rows
    that are almost entirely near-black (a uniform border row) versus rows
    with real image content. This matters because the strip's outer scan
    border is often directly adjacent to -- and merges visually with -- the
    top and bottom channel's own internal separator border, which would
    throw off a naive equal-thirds split by a few rows. A brightness-mean
    test alone is unreliable (a real photo can have full-width dark rows),
    so this instead requires almost every pixel in the row to be near-black
    before calling it a border row. Falls back to equal thirds if exactly
    three content bands can't be found this way.

    `dark_threshold`: a pixel below this value counts as "black".
    `dark_frac_cutoff`: a row counts as border if at least this fraction of
    its pixels are black. Lower this (e.g. to 0.8) if borders are cropping
    too little; raise it if too much real content is getting cropped.
    """
    h, w = img.shape
    dark_frac = np.mean(img < dark_threshold, axis=1)
    is_content = dark_frac < dark_frac_cutoff

    runs = []
    start = None
    for i, bright in enumerate(is_content):
        if bright and start is None:
            start = i
        elif not bright and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, h - 1))

    # keep the three largest bright runs, in top-to-bottom order
    runs = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:3]
    runs = sorted(runs, key=lambda r: r[0])

    if len(runs) != 3:
        third = h // 3
        top = img[0 * third:1 * third, :]
        mid = img[1 * third:2 * third, :]
        bot = img[2 * third:3 * third, :]
        min_h = min(top.shape[0], mid.shape[0], bot.shape[0])
        return top[:min_h], mid[:min_h], bot[:min_h]

    blocks = [img[r0:r1 + 1, :] for r0, r1 in runs]
    min_h = min(b.shape[0] for b in blocks)
    blocks = [b[:min_h] for b in blocks]
    return blocks[0], blocks[1], blocks[2]


# --------------------------------------------------------------------------
# Step 4: remove any additional bordering effects on each channel
# --------------------------------------------------------------------------
#
# IMPORTANT: this must crop all three channels by the SAME amount on each
# side. Cropping each channel independently based on its own content would
# shift each channel's pixel coordinate origin by a different amount,
# silently corrupting the coordinate frame that the anchor points and
# cross-correlation search below both depend on being shared across R/G/B.

def remove_border_effects(images, trim_frac=0.02, threshold=15):
    """Given the three channel images (same shape), trim a small fixed
    fraction off every edge (removes sprocket-hole / scan-artifact borders),
    then crop all three to the intersection of their non-black bounding
    boxes -- applying one shared crop to every channel so their coordinate
    frames stay in sync.
    """
    h, w = images[0].shape
    bh = max(1, int(round(h * trim_frac))) if trim_frac > 0 else 0
    bw = max(1, int(round(w * trim_frac))) if trim_frac > 0 else 0
    if bh > 0 and bw > 0 and h - 2 * bh > 1 and w - 2 * bw > 1:
        images = [img[bh:h - bh, bw:w - bw] for img in images]

    h2, w2 = images[0].shape
    rmins, rmaxs, cmins, cmaxs = [], [], [], []
    for img in images:
        mask = img > threshold
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            rmins.append(0); rmaxs.append(h2 - 1)
            cmins.append(0); cmaxs.append(w2 - 1)
            continue
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        rmins.append(r0); rmaxs.append(r1)
        cmins.append(c0); cmaxs.append(c1)

    r0, r1 = max(rmins), min(rmaxs)
    c0, c1 = max(cmins), min(cmaxs)
    if r1 <= r0 or c1 <= c0:
        return images  # nothing sensible to crop further
    return [img[r0:r1 + 1, c0:c1 + 1] for img in images]


# --------------------------------------------------------------------------
# Step 5: display with scale bars
# --------------------------------------------------------------------------

def display_with_scale_bars(images, titles, block=True):
    """Show the images side by side with pixel-coordinate axes/scale bars."""
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5.5))
    if len(images) == 1:
        axes = [axes]
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.tick_params(labelbottom=True, labelleft=True)
    fig.suptitle("Channel images with pixel scale bars")
    plt.tight_layout()
    plt.show(block=block)
    return fig


# --------------------------------------------------------------------------
# Step 6: interactive anchor-point selection
# --------------------------------------------------------------------------

def get_anchor_points(images, titles, n_anchors=1):
    """Ask the user to click `n_anchors` corresponding point(s) in each
    image (same real-world feature, clicked in the same order in every
    image). Returns a list (per image) of lists of (x, y) tuples.
    """
    all_points = []
    for im, title in zip(images, titles):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(im, cmap="gray")
        ax.set_title(
            f"{title}: click {n_anchors} anchor point(s), in the same "
            f"order as the other images"
        )
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        pts = plt.ginput(n_anchors, timeout=0)
        plt.close(fig)
        if len(pts) < n_anchors:
            raise RuntimeError(
                f"Only {len(pts)}/{n_anchors} anchor point(s) were selected "
                f"on '{title}'. Please select all required points."
            )
        all_points.append(pts)
    return all_points


# --------------------------------------------------------------------------
# Step 7: anchor-seeded, difference-minimizing alignment
# --------------------------------------------------------------------------

def shift_image(img, dy, dx, fill=0):
    """Translate `img` by (dy, dx), filling exposed areas with `fill`."""
    h, w = img.shape
    out = np.full_like(img, fill)
    src_y0, src_y1 = max(0, -dy), h - max(0, dy)
    src_x0, src_x1 = max(0, -dx), w - max(0, dx)
    dst_y0, dst_y1 = max(0, dy), h - max(0, -dy)
    dst_x0, dst_x1 = max(0, dx), w - max(0, -dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def _patch_around(img, cy, cx, half):
    """Extract a square patch of half-width `half` centered at (cy, cx),
    clipped to image bounds.
    """
    h, w = img.shape
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    return img[y0:y1, x0:x1], (y0, x0)


def difference_score(a, b):
    """Sum-of-squared-differences between two equally-shaped patches.
    Minimizing this is equivalent to maximizing the (zero-lag) cross
    correlation between the patches once their means are removed:
        SSD(a,b) = sum(a^2) + sum(b^2) - 2 * crosscorr(a,b)
    so searching for the minimum of SSD over candidate shifts is the
    same search as maximizing normalized cross-correlation.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.mean((a - b) ** 2)


def align_via_local_search(ref, mov, initial_shift, search_range=15,
                            patch_half=60, anchor_xy=None):
    """Find the (dy, dx) shift to apply to `mov` that best aligns it with
    `ref`, by exhaustively searching a window around `initial_shift` and
    minimizing the sum-of-squared-differences function (equivalent to
    maximizing local cross-correlation).

    The search is evaluated on a patch cropped around the anchor point (if
    given) or the image center, both for speed and so that the score isn't
    dominated by unrelated parts of a large image.
    """
    h, w = ref.shape
    if anchor_xy is not None:
        cx, cy = int(round(anchor_xy[0])), int(round(anchor_xy[1]))
    else:
        cy, cx = h // 2, w // 2

    ref_patch, (py0, px0) = _patch_around(ref, cy, cx, patch_half)
    ph, pw = ref_patch.shape

    dy0, dx0 = initial_shift
    best_shift = (dy0, dx0)
    best_score = np.inf

    for dy in range(dy0 - search_range, dy0 + search_range + 1):
        for dx in range(dx0 - search_range, dx0 + search_range + 1):
            # sample the same patch location out of a shifted `mov`
            my0, mx0 = py0 - dy, px0 - dx
            my1, mx1 = my0 + ph, mx0 + pw
            if my0 < 0 or mx0 < 0 or my1 > h or mx1 > w:
                continue
            mov_patch = mov[my0:my1, mx0:mx1]
            score = difference_score(ref_patch, mov_patch)
            if score < best_score:
                best_score = score
                best_shift = (dy, dx)

    return best_shift, best_score


# --------------------------------------------------------------------------
# Step 8: crop composite to the common (triple-covered) region
# --------------------------------------------------------------------------

def crop_common_region(r_shift, g_shift, b_shift, shape):
    """Given the (dy, dx) shifts applied to bring G and B onto R (R itself
    is shift (0,0)), compute the sub-rectangle of `shape` that is covered
    by valid (non-fill) pixels in all three shifted channels, and return
    the crop slice (y0, y1, x0, x1).
    """
    h, w = shape
    y0 = max(0, r_shift[0], g_shift[0], b_shift[0])
    y1 = h + min(0, r_shift[0], g_shift[0], b_shift[0])
    x0 = max(0, r_shift[1], g_shift[1], b_shift[1])
    x1 = w + min(0, r_shift[1], g_shift[1], b_shift[1])
    y0, y1 = max(0, y0), max(y0, y1)
    x0, x1 = max(0, x0), max(x0, x1)
    return y0, y1, x0, x1


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the input strip image")
    parser.add_argument("--output-dir", default="output",
                         help="Directory to write output images (default: output/)")
    parser.add_argument("--n-anchors", type=int, default=1,
                         help="Number of anchor points to click per image (default: 1)")
    parser.add_argument("--search-range", type=int, default=15,
                         help="+/- pixel search window around the anchor-based "
                              "initial guess (default: 15)")
    parser.add_argument("--patch-half", type=int, default=60,
                         help="Half-width of the patch used to score alignment "
                              "(default: 60)")
    parser.add_argument("--no-display", action="store_true",
                         help="Skip the interactive scale-bar preview window")
    parser.add_argument("--manual-crop", action="store_true",
                         help="Drag-select crop rectangles by hand instead of "
                              "relying on automatic black-border detection")
    parser.add_argument("--edge-threshold", type=int, default=10,
                         help="Pixel value below which a pixel counts as "
                              "'black' for the outer-edge crop (default: 10)")
    parser.add_argument("--split-dark-threshold", type=int, default=10,
                         help="Pixel value below which a pixel counts as "
                              "'black' when detecting channel-band borders "
                              "during splitting (default: 10)")
    parser.add_argument("--split-dark-frac", type=float, default=0.9,
                         help="Fraction of a row that must be black for it "
                              "to be treated as a border row during "
                              "splitting; lower this if too little border "
                              "gets removed, raise it if too much content "
                              "gets cropped (default: 0.9)")
    parser.add_argument("--residual-threshold", type=int, default=15,
                         help="Pixel value below which a pixel counts as "
                              "'black' when trimming leftover per-channel "
                              "border artifacts (default: 15)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1-2. load + grayscale + remove outer black edges
    print("Loading and converting to 8-bit grayscale...")
    gray = load_grayscale(args.input)
    if args.manual_crop:
        print("Drag a box around the full three-channel strip (excluding the outer border)...")
        gray = manual_crop(gray, "Select the full strip")
    else:
        gray = remove_black_edges(gray, threshold=args.edge_threshold)

    # 3. split into three channel images
    print("Splitting vertical strip into three channels...")
    top, mid, bot = split_vertical_strip(
        gray,
        dark_threshold=args.split_dark_threshold,
        dark_frac_cutoff=args.split_dark_frac,
    )
    # Standard Prokudin-Gorskii order top->bottom is Blue, Green, Red.
    imageB, imageG, imageR = top, mid, bot

    # 4. remove any extra border artifacts, cropping all three channels by
    #    the same amount so their coordinate frames stay in sync
    if args.manual_crop:
        print("Drag a box on the R channel to trim any remaining border "
              "(the same crop is applied to G and B)...")
        imageR, imageG, imageB = manual_crop_shared(
            [imageR, imageG, imageB], "Trim remaining border"
        )
    else:
        imageR, imageG, imageB = remove_border_effects(
            [imageR, imageG, imageB], threshold=args.residual_threshold
        )

    r_path = os.path.join(args.output_dir, "imageR.png")
    g_path = os.path.join(args.output_dir, "imageG.png")
    b_path = os.path.join(args.output_dir, "imageB.png")
    Image.fromarray(imageR).save(r_path)
    Image.fromarray(imageG).save(g_path)
    Image.fromarray(imageB).save(b_path)
    print(f"Saved channel images: {r_path}, {g_path}, {b_path}")

    channels = [imageR, imageG, imageB]
    titles = ["R", "G", "B"]

    # 5. display with scale bars
    if not args.no_display:
        print("Displaying channels with pixel scale bars...")
        display_with_scale_bars(channels, titles, block=True)

    # 6. anchor points
    print(f"Click {args.n_anchors} corresponding anchor point(s) in each "
          f"image window, in the same order each time.")
    anchor_points = get_anchor_points(channels, titles, n_anchors=args.n_anchors)
    r_anchors, g_anchors, b_anchors = anchor_points

    # Use the first anchor pair to seed the initial guess; average over all
    # anchors if more than one was collected.
    r_pts = np.array(r_anchors)
    g_pts = np.array(g_anchors)
    b_pts = np.array(b_anchors)

    # anchor points are (x, y); shifts are (dy, dx) needed to move mov -> ref
    g_init = np.mean(r_pts - g_pts, axis=0)  # (dx, dy) diff, avg over anchors
    b_init = np.mean(r_pts - b_pts, axis=0)
    g_init_shift = (int(round(g_init[1])), int(round(g_init[0])))  # (dy, dx)
    b_init_shift = (int(round(b_init[1])), int(round(b_init[0])))

    ref_anchor_xy = tuple(np.mean(r_pts, axis=0))

    # 7. refine with local difference-minimizing / cross-correlation search
    print("Refining alignment via local cross-correlation search...")
    g_shift, g_score = align_via_local_search(
        imageR, imageG, g_init_shift,
        search_range=args.search_range, patch_half=args.patch_half,
        anchor_xy=ref_anchor_xy,
    )
    b_shift, b_score = align_via_local_search(
        imageR, imageB, b_init_shift,
        search_range=args.search_range, patch_half=args.patch_half,
        anchor_xy=ref_anchor_xy,
    )
    print(f"  G shift: {g_shift} (anchor guess was {g_init_shift}), "
          f"final SSD={g_score:.3f}")
    print(f"  B shift: {b_shift} (anchor guess was {b_init_shift}), "
          f"final SSD={b_score:.3f}")

    # 8. build the aligned composite and crop to the common region
    r_shift = (0, 0)
    imageG_aligned = shift_image(imageG, *g_shift)
    imageB_aligned = shift_image(imageB, *b_shift)

    y0, y1, x0, x1 = crop_common_region(r_shift, g_shift, b_shift, imageR.shape)
    if y1 <= y0 or x1 <= x0:
        print("Warning: computed shifts leave no common region; skipping crop.")
        y0, y1, x0, x1 = 0, imageR.shape[0], 0, imageR.shape[1]

    composite = np.dstack([
        imageR[y0:y1, x0:x1],
        imageG_aligned[y0:y1, x0:x1],
        imageB_aligned[y0:y1, x0:x1],
    ])

    out_path = os.path.join(args.output_dir, "composite_aligned.png")
    Image.fromarray(composite, mode="RGB").save(out_path)
    print(f"Saved aligned composite: {out_path}")

    if not args.no_display:
        plt.figure(figsize=(8, 8))
        plt.imshow(composite)
        plt.title("Aligned RGB composite")
        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()