"""
End-to-end test of the Block 1 pipeline.

Loads the GME/Terna data from `data/`, computes empirical forecast-error
residuals on the MGP vs MI overlap, fits the four candidate distributions
separately for the gas-crisis-2022 and normalized-2023-2024 regimes, and
emits:

    - calibrations.json    -> all fitted parameters per regime
    - calibration_table.csv-> publication-style table (Table 1 of the paper)
    - forecast_error_calibration.png/.pdf -> diagnostic figure (Figure 3)

Run from the repo root with:

    python tests_block1.py --data-dir data --output-dir results/block1

Defaults work out-of-the-box if both directories exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from market_data_loader import MarketDataLoader
from forecast_error_calibration import (ForecastErrorCalibrator,
                                        default_regime_assignment,
                                        calibrations_to_table,
                                        save_calibrations_json)
from forecast_error_plots import plot_calibration_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data',
                    help='Directory with GME/Terna XLSX files')
    ap.add_argument('--output-dir', default='results/block1',
                    help='Where to write JSON, CSV, and figures')
    ap.add_argument('--msd-zone', default='Nord',
                    help='MSD zone filename suffix (Nord, CNor, CSud, Sud, ...)')
    ap.add_argument('--min-mi-price', type=float, default=5.0,
                    help='Drop residual rows where MI < this threshold (EUR/MWh)')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("[1/4] Loading market data ...")
    loader = MarketDataLoader(args.data_dir, msd_zone=args.msd_zone)
    data = loader.load_all()
    print(data.coverage_report())

    if data.pun.empty or data.mi.empty:
        print("ERROR: PUN or MI data missing; nothing to calibrate.",
              file=sys.stderr)
        sys.exit(1)

    # 2. Calibrate
    print("\n[2/4] Calibrating forecast-error distributions ...")
    calib = ForecastErrorCalibrator(min_price_eur_mwh=args.min_mi_price)
    cals = calib.fit_by_regime(data.merged, default_regime_assignment)
    for name, cal in cals.items():
        print(f"  {name}: n={cal.n_hours}, std={cal.std:.4f}, "
              f"lag1={cal.lag1_autocorr:.3f}")

    # 3. Save table + JSON
    print("\n[3/4] Saving calibration table and JSON ...")
    table = calibrations_to_table(cals)
    csv_path = out_dir / 'calibration_table.csv'
    table.to_csv(csv_path)
    print(f"  -> {csv_path}")
    json_path = out_dir / 'calibrations.json'
    save_calibrations_json(cals, str(json_path))
    print(f"  -> {json_path}")

    # 4. Plot
    print("\n[4/4] Producing diagnostic figure ...")
    # Recompute residuals split by regime for the plotting (the calibrator
    # already did this internally; we redo it here cheaply).
    residuals = calib.compute_residuals(data.merged)
    labels = default_regime_assignment(residuals.index)
    residuals = residuals.assign(regime=labels.values).dropna(subset=['regime'])
    residuals_by_regime = {
        name: grp['err_rel'].to_numpy()
        for name, grp in residuals.groupby('regime', sort=False)
    }
    fig_path = plot_calibration_panel(
        residuals_by_regime, cals,
        output_path=out_dir / 'forecast_error_calibration.png')
    print(f"  -> {fig_path}")
    print(f"  -> {fig_path.with_suffix('.pdf')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
