"""
He red line (667.8 nm) Zeeman analysis — anomalous Zeeman effect.

Expects paired sigma/pi images per VDC:
  vdc0_He_red_667_sigma.png, vdc0_He_red_667_pi.png
  vdc5_He_red_667_sigma.png, vdc5_He_red_667_pi.png
  ... (same for 10, 15, 20, 25, 30)

VDC → B (kG): 0→0.2, 5→1.5, 10→2.97, 15→4.02, 20→5.97, 25→7.35, 30→9.3

Workflow:
  1. Preprocess (if needed): python zeeman_red.py --preprocess -d <data_dir>
  2. Analyze: python zeeman_red.py -o red_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Reuse analysis from yellow (same sigma/pi logic)
from zeeman_yellow import (
    VDC_TO_B_KG,
    VDC_VALUES,
    B_FIELDS_KG,
    load_cached_profiles,
    run_yellow_analysis,
    plot_results,
)

# For preprocessing
from zeeman_analysis_pipeline import load_or_compute_profiles


def main():
    parser = argparse.ArgumentParser(description="He red line (667.8 nm) Zeeman analysis — anomalous Zeeman")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess images and save cache (run once)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("red_cache"),
                        help="Cache/output directory (default: red_cache)")
    parser.add_argument("-d", "--data-dir", type=Path, default=Path("."),
                        help="Directory with vdc0_He_red_667_sigma.png, vdc0_He_red_667_pi.png, etc.")
    parser.add_argument("images", nargs="*", type=Path,
                        help="Explicit paths: sigma0, pi0, sigma5, pi5, ... Overrides auto-discovery.")
    parser.add_argument("--etalon-cm", type=float, default=0.811, help="Etalon thickness (cm)")
    parser.add_argument("-g", "--g-effective", type=float, default=1.0,
                        help="Effective Landé g for μ_B extraction")
    parser.add_argument("-B", "--B-fields", type=float, nargs="+", default=B_FIELDS_KG,
                        help="B fields in kG, in image order")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plot")
    parser.add_argument("--precise", action="store_true", help="Slower center search (step=1)")
    args = parser.parse_args()

    cache_dir = args.output_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.preprocess:
        # Build image list: (sigma, pi) pairs per VDC
        if args.images:
            image_paths = [Path(p).resolve() for p in args.images]
        else:
            image_paths = []
            for vdc in VDC_VALUES:
                base = f"vdc{vdc}_He_red_667"
                sigma_p = args.data_dir / f"{base}_sigma.png"
                pi_p = args.data_dir / f"{base}_pi.png"
                if not sigma_p.exists() or not pi_p.exists():
                    raise FileNotFoundError(
                        f"For VDC={vdc}, expected {base}_sigma.png and {base}_pi.png in {args.data_dir}"
                    )
                image_paths.extend([sigma_p, pi_p])
        print("Preprocessing He red images...")
        load_or_compute_profiles(
            image_paths=[str(p) for p in image_paths],
            cache_dir=cache_dir,
            force_recompute=True,
            nbins=2000,
            verbose=True,
            center_step=1 if args.precise else 2,
        )
        print(f"Cached to {cache_dir}")
        print("Run without --preprocess to analyze.")
        return

    # Infer n_vdc from cache (2 profiles per VDC: sigma, pi)
    r_list, _, _ = load_cached_profiles(cache_dir)
    n_vdc = len(r_list) // 2
    B_kG = args.B_fields[:n_vdc] if len(args.B_fields) >= n_vdc else B_FIELDS_KG[:n_vdc]

    df_ring, df_summary, magneton = run_yellow_analysis(
        cache_dir=cache_dir,
        B_fields_kG=B_kG,
        etalon_thickness_cm=args.etalon_cm,
        g_effective=args.g_effective,
    )

    df_ring.to_csv(cache_dir / "zeeman_red_ringwise.csv", index=False)
    df_summary.to_csv(cache_dir / "zeeman_red_summary.csv", index=False)
    print(f"Saved to {cache_dir}")

    if not args.no_plot:
        plot_results(df_summary, magneton, cache_dir / "zeeman_red_vs_B.png", line_label="He red line")

    mu, mu_u = magneton["mu_B_J_per_T"], magneton.get("mu_B_unc_J_per_T")
    mu_str = f"{mu:.4e}" + (f" ± {mu_u:.2e}" if mu_u else "")
    print()
    print("He red line (anomalous Zeeman, sigma/pi):")
    print(f"  Estimated μ_B = {mu_str} J/T (g_eff={args.g_effective})")
    print(f"  Accepted μ_B  = {magneton['accepted_mu_B']:.4e} J/T")


if __name__ == "__main__":
    main()
