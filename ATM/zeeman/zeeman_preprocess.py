"""
Zeeman Preprocessing: Run ONCE to compute and cache radial profiles from Fabry-Perot images.

This script performs the expensive steps:
  - Load images, subtract background
  - Find interferometer center (ring symmetry refinement)
  - Compute radial profiles I(r) for each image

Output (saved to cache_dir):
  - r_arr.npy      : radius arrays, one per image
  - I_r_arr.npy    : radial intensity arrays
  - centers.npy    : (x, y) center for each image

After running once, use zeeman_analyze.py to load this cache and compute
fractional frequency shifts + Bohr magneton without re-running preprocessing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zeeman_analysis_pipeline import load_or_compute_profiles


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Fabry-Perot images and cache radial profiles."
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Paths to splitting measurement images (PNG, etc.), in order of B field.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save r_arr.npy, I_r_arr.npy, centers.npy (default: current dir)",
    )
    parser.add_argument(
        "-n", "--nbins",
        type=int,
        default=2000,
        help="Number of radial bins (default: 2000)",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force recompute even if cache exists",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster preprocessing (center_step=2, ~2–3x speedup)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress for each image",
    )
    args = parser.parse_args()

    # Expand globs (e.g. *.png) — Windows doesn't expand these in the shell
    image_paths = []
    for p in args.images:
        p = Path(p)
        if "*" in str(p) or "?" in str(p):
            expanded = sorted(p.parent.glob(p.name))
            if not expanded:
                raise FileNotFoundError(f"No files match: {p}")
            image_paths.extend(expanded)
        else:
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            image_paths.append(p)

    for p in image_paths:
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")

    I_r_arr, r_arr, centers = load_or_compute_profiles(
        image_paths=[str(p) for p in image_paths],
        cache_dir=args.output_dir,
        force_recompute=args.force,
        nbins=args.nbins,
        verbose=args.verbose,
        center_step=2 if args.fast else 1,
    )

    print(f"Saved cached profiles to {args.output_dir.absolute()}")
    print(f"  - r_arr.npy     : {len(r_arr)} profiles")
    print(f"  - I_r_arr.npy   : {len(I_r_arr)} profiles")
    print(f"  - centers.npy   : {centers.shape}")
    print("\nNext: run zeeman_analyze.py (or use zeeman_analysis_quick.ipynb)")


if __name__ == "__main__":
    main()
