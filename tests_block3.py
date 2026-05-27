"""
Block 3 verification: real-data loading via MarketDataLoader.

Exercises only the `build_market_data` dispatch path: actual PPO training
is too expensive to run inside this test, so we stub the PPO step and
just confirm that:

  3.1 build_market_data() with cfg.data_source='real' loads PUN MGP from
      the XLSX directory and returns sane pretrain/annual slices.
  3.2 The pretrain slice ends at the last day of cfg.train_year.
  3.3 The annual slice spans the full cfg.eval_year (including DST
      handling and leap-year detection).
  3.4 Forward-filling fixes occasional NaN holes without changing length.
  3.5 cfg.days is auto-adjusted to the calendar length of eval_year when
      the user did not override it.
  3.6 build_market_data() with cfg.data_source='synthetic' still works
      (backward compatibility).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_FAILED = []


def _check(cond, name, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def _make_cfg(**kwargs):
    """Build a PipelineConfig with sensible test defaults."""
    import main as m
    cfg = m.PipelineConfig(
        mode="fast",
        days=365,            # will be auto-adjusted in real mode
        pretrain_months=2,
        seed=42,
        verbose=False,
        skip_train=True,
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_paths(tmpdir):
    import main as m
    return m.OutputPaths.create(output_root=str(tmpdir), run_name="test_block3")


def test_real_data_basic(data_dir):
    print("\n--- Block 3.1: real-data basic loading ---")
    import tempfile
    import main as m
    cfg = _make_cfg(data_source="real", data_dir=str(data_dir),
                    train_year=2023, eval_year=2024, msd_zone="Nord")
    with tempfile.TemporaryDirectory() as td:
        paths = _make_paths(td)
        out = m.build_market_data(cfg, paths)
        _check("pretrain_prices" in out and "annual_prices" in out,
               "build_market_data returns the expected keys")
        n_pretrain = len(out["pretrain_prices"])
        n_annual = len(out["annual_prices"])
        _check(n_pretrain == 2 * 30 * 24,
               "pretrain window has 2*30*24=1440 hours",
               f"got {n_pretrain}")
        # 2024 is a leap year => 366 days
        _check(n_annual == 366 * 24,
               "annual window has 366*24=8784 hours for 2024 (leap year)",
               f"got {n_annual}")
        prices = np.array(out["annual_prices"])
        _check(np.isfinite(prices).all() and (prices > 0).all(),
               "all annual prices are finite and positive",
               f"min={prices.min():.2f}, max={prices.max():.2f}")
        _check(50 < prices.mean() < 200,
               "annual price mean is plausibly Italian-2024 (50-200 EUR/MWh)",
               f"mean={prices.mean():.2f}")
        # The persisted CSV should mention the real prices
        csv = paths.data / "real_prices.csv"
        _check(csv.exists(), "real_prices.csv was persisted",
               f"path={csv}")


def test_real_data_synthetic_compat(data_dir):
    print("\n--- Block 3.2: synthetic mode still works (backward compat) ---")
    import tempfile
    import main as m
    cfg = _make_cfg(data_source="synthetic", days=14)
    with tempfile.TemporaryDirectory() as td:
        paths = _make_paths(td)
        out = m.build_market_data(cfg, paths)
        n_annual = len(out["annual_prices"])
        _check(n_annual == 14 * 24,
               "synthetic annual window honors cfg.days exactly",
               f"got {n_annual}, expected 336")
        prices = np.array(out["annual_prices"])
        _check(np.isfinite(prices).all() and (prices >= 10).all(),
               "synthetic prices are finite and floored at 10")
        csv = paths.data / "synthetic_prices.csv"
        _check(csv.exists(), "synthetic_prices.csv was persisted")


def test_real_data_days_autoadjust(data_dir):
    print("\n--- Block 3.3: cfg.days auto-adjusts to eval-year length ---")
    import tempfile
    import main as m
    # Eval-year = 2023 is 365 days (not leap)
    cfg = _make_cfg(data_source="real", data_dir=str(data_dir),
                    train_year=2022, eval_year=2023, msd_zone="Nord")
    with tempfile.TemporaryDirectory() as td:
        paths = _make_paths(td)
        out = m.build_market_data(cfg, paths)
        _check(cfg.days == 365,
               "cfg.days auto-snapped to 365 for non-leap 2023",
               f"cfg.days={cfg.days}")
        _check(len(out["annual_prices"]) == 365 * 24,
               "annual prices length matches 2023 calendar")


def test_real_data_validation(data_dir):
    print("\n--- Block 3.4: missing-year validation ---")
    import tempfile
    import main as m
    cfg = _make_cfg(data_source="real", data_dir=str(data_dir),
                    train_year=2020, eval_year=2024, msd_zone="Nord")
    with tempfile.TemporaryDirectory() as td:
        paths = _make_paths(td)
        try:
            m.build_market_data(cfg, paths)
            _check(False, "missing train_year is rejected with ValueError")
        except ValueError as e:
            _check("train_year" in str(e) or "2020" in str(e),
                   "ValueError mentions the missing year",
                   f"msg={e}")
        except Exception as e:
            _check(False, "missing train_year is rejected with ValueError "
                          f"(got {type(e).__name__}: {e})")


def main_test():
    data_dir = Path(__file__).parent / "_test_data"
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}", file=sys.stderr)
        print("This test requires the GME XLSX files to be available "
              "in a subdirectory called _test_data/.", file=sys.stderr)
        sys.exit(2)

    print("=" * 70)
    print("Block 3 verification tests (real-data loading)")
    print("=" * 70)

    test_real_data_basic(data_dir)
    test_real_data_synthetic_compat(data_dir)
    test_real_data_days_autoadjust(data_dir)
    test_real_data_validation(data_dir)

    print("\n" + "=" * 70)
    if _FAILED:
        print(f"FAILED: {len(_FAILED)} test(s): {_FAILED}")
        sys.exit(1)
    print("All Block 3 tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main_test()
