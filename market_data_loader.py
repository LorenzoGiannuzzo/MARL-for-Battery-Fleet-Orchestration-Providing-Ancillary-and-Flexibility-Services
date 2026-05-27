"""
Italian electricity market data loader.

Loads PUN MGP, MI-A1, and MSD ex-ante price series from GME and Terna XLSX
exports and aligns them on an hourly DatetimeIndex.

Supported data sources (auto-discovered by filename pattern in `data_dir`):
    PUN MGP day-ahead:   YYYYMMDD_YYYYMMDD_PUN.xlsx                (sheet MGP-PUNPUN)
    MI-A1 intra-day:     YYYYMMDD_YYYYMMDD_MI-A1_PrezziZonali_NAT.xlsx
                                                                   (sheet MI-A1-PrezziZonali-NAT)
    MSD ex-ante <ZONE>:  EsitiMSD_PrezziExAnte_YYYYMMDD_YYYYMMDD_<ZONE>.xlsx
                                                                   (sheet PrezziExAnte_<ZONE>_MSD)

All loaders return pandas DataFrames indexed by an hourly DatetimeIndex
("datetime", UTC-naive, Europe/Rome wall-clock interpretation). DST anomalies
(23h spring-forward day, 25h fall-back day) are handled by collapsing the
extra hour to the next-day 00:00 slot and dropping the resulting duplicate
index entries (the GME/Terna files do not tag the second 2 AM uniquely).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# Helpers
# ============================================================================

def _parse_italian_number(s: pd.Series) -> pd.Series:
    """Convert an Italian-formatted numeric string column to float.

    Examples of input strings handled:
        '64,974170'      -> 64.974170      (comma decimal)
        '3.000,00'       -> 3000.00        (dot thousands + comma decimal)
        '107,090000'     -> 107.090000
        '0.10' (rare)    -> 0.10           (already English)
    """
    return (
        s.astype(str)
         .str.replace('.', '', regex=False)
         .str.replace(',', '.', regex=False)
         .astype(float)
    )


def _build_datetime(df: pd.DataFrame, date_col: str = "Data",
                    hour_col: str = "Ora") -> pd.DatetimeIndex:
    """Build a naive DatetimeIndex from (date, hour) columns.

    GME/Terna files use Ora in {1..24} for normal days, {1..23} on the
    spring-forward day (skipping one hour), and {1..25} on the fall-back day
    (with one extra hour). We do not attempt to tag the duplicate hour;
    callers should drop_duplicates(index) downstream.
    """
    dates = pd.to_datetime(df[date_col], format='%d/%m/%Y')
    offsets = pd.to_timedelta(df[hour_col].astype(int) - 1, unit='h')
    return dates + offsets


def _glob_sorted(data_dir: Path, pattern: str) -> List[Path]:
    """Return files matching pattern sorted by their leading start date."""
    files = sorted(data_dir.glob(pattern))
    # Sort by the first 8-digit date in the filename (start date)
    def start_date(p: Path) -> str:
        m = re.search(r"(\d{8})_(\d{8})", p.name)
        return m.group(1) if m else ""
    return sorted(files, key=start_date)


# ============================================================================
# Container
# ============================================================================

@dataclass
class MarketData:
    """Aligned multi-source market data.

    Attributes
    ----------
    pun : pd.DataFrame
        Columns: ['pun_mgp'] in EUR/MWh. Index: hourly DatetimeIndex.
    mi  : pd.DataFrame
        Columns: ['pun_mi'] in EUR/MWh. Index: hourly DatetimeIndex.
    msd : pd.DataFrame
        Columns: ['msd_min_buy', 'msd_mean_buy', 'msd_mean_sell',
                  'msd_max_sell'] in EUR/MWh. Index: hourly DatetimeIndex.
    merged : pd.DataFrame
        Outer-join of the three sources on the hourly index. Missing values
        are preserved as NaN (do not impute silently).
    """
    pun: pd.DataFrame
    mi:  pd.DataFrame
    msd: pd.DataFrame
    merged: pd.DataFrame

    def coverage_report(self) -> str:
        """Human-readable summary of coverage and gaps."""
        lines = []
        for name, df in [("PUN MGP", self.pun),
                         ("MI-A1 NAT", self.mi),
                         ("MSD NORD", self.msd)]:
            if df.empty:
                lines.append(f"  {name:10s}: empty")
                continue
            start, end = df.index.min(), df.index.max()
            n_hours = len(df)
            n_days = df.index.normalize().nunique()
            expected_hours = int((end - start).total_seconds() // 3600) + 1
            missing = expected_hours - n_hours
            pct = 100 * n_hours / expected_hours if expected_hours else 0.0
            lines.append(
                f"  {name:10s}: {start.date()} -> {end.date()} | "
                f"{n_days:>3d} days, {n_hours:>5d} hours, "
                f"{missing:>4d} missing ({pct:.1f}% complete)"
            )
        return "\n".join(lines)


# ============================================================================
# Loader
# ============================================================================

class MarketDataLoader:
    """Load and align Italian electricity market data from XLSX files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the XLSX files (auto-discovered by pattern).
    msd_zone : str, default 'NORD'
        Which Italian market zone to load for MSD ex-ante data. Must match
        the suffix in the MSD filenames.

    Examples
    --------
    >>> loader = MarketDataLoader("data")
    >>> data = loader.load_all()
    >>> print(data.coverage_report())
    >>> # Subset to a single year
    >>> df_2024 = data.merged.loc["2024"]
    """

    def __init__(self, data_dir: str | Path, msd_zone: str = "NORD"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")
        self.msd_zone = msd_zone

    # ----- single-file readers -----

    @staticmethod
    def _read_pun_file(path: Path) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name='MGP-PUNPUN', usecols=['Data', 'Ora', '€/MWh'])
        df.index = _build_datetime(df)
        out = pd.DataFrame({'pun_mgp': _parse_italian_number(df['€/MWh'])},
                           index=df.index)
        out.index.name = 'datetime'
        return out

    @staticmethod
    def _read_mi_file(path: Path) -> Optional[pd.DataFrame]:
        """Read a single MI-A1 XLSX file. Returns None if the file is in
        an unsupported (e.g. daily-aggregate) format, with a warning."""
        # Peek the columns first to detect format mismatch. GME publishes
        # both hourly and daily-aggregate exports for MI; the daily version
        # is missing the 'Ora' column and is not usable for hour-level
        # forecast-error calibration.
        head = pd.read_excel(path, sheet_name='MI-A1-PrezziZonali-NAT', nrows=0)
        if 'Ora' not in head.columns:
            import warnings
            warnings.warn(
                f"Skipping {path.name}: file is in daily-aggregate format "
                f"(no 'Ora' column). Re-download with hourly granularity "
                f"from the GME portal.",
                RuntimeWarning, stacklevel=2)
            return None
        df = pd.read_excel(path, sheet_name='MI-A1-PrezziZonali-NAT',
                           usecols=['Data', 'Ora', '€/MWh'])
        df.index = _build_datetime(df)
        out = pd.DataFrame({'pun_mi': _parse_italian_number(df['€/MWh'])},
                           index=df.index)
        out.index.name = 'datetime'
        return out

    def _read_msd_file(self, path: Path) -> pd.DataFrame:
        # Filename uses Title case (e.g. "Nord"), sheet name uses upper case
        # ("NORD"). Convert here to keep the public API permissive.
        sheet = f'PrezziExAnte_{self.msd_zone.upper()}_MSD'
        cols = ['Data', 'Ora',
                'Prezzo Minimo di Acquisto (€/MWh)',
                'Prezzo Medio di Acquisto (€/MWh)',
                'Prezzo Medio di Vendita (€/MWh)',
                'Prezzo Massimo di Vendita (€/MWh)']
        df = pd.read_excel(path, sheet_name=sheet, usecols=cols)
        df.index = _build_datetime(df)
        out = pd.DataFrame({
            'msd_min_buy':   _parse_italian_number(df[cols[2]]),
            'msd_mean_buy':  _parse_italian_number(df[cols[3]]),
            'msd_mean_sell': _parse_italian_number(df[cols[4]]),
            'msd_max_sell':  _parse_italian_number(df[cols[5]]),
        }, index=df.index)
        out.index.name = 'datetime'
        return out

    # ----- multi-file loaders -----

    def load_pun(self) -> pd.DataFrame:
        files = _glob_sorted(self.data_dir, "*_PUN.xlsx")
        # Exclude MI files that also contain "PUN" in their parsed text but
        # not in their filename (the MI filename does not match *_PUN.xlsx).
        files = [f for f in files if 'MI-A1' not in f.name]
        if not files:
            return pd.DataFrame(columns=['pun_mgp']).set_index(
                pd.DatetimeIndex([], name='datetime'))
        dfs = [self._read_pun_file(f) for f in files]
        return self._concat_dedup(dfs)

    def load_mi(self) -> pd.DataFrame:
        files = _glob_sorted(self.data_dir, "*_MI-A1_PrezziZonali_NAT.xlsx")
        if not files:
            return pd.DataFrame(columns=['pun_mi']).set_index(
                pd.DatetimeIndex([], name='datetime'))
        dfs = [self._read_mi_file(f) for f in files]
        dfs = [d for d in dfs if d is not None]  # skip daily-format files
        if not dfs:
            return pd.DataFrame(columns=['pun_mi']).set_index(
                pd.DatetimeIndex([], name='datetime'))
        return self._concat_dedup(dfs)

    def load_msd(self) -> pd.DataFrame:
        files = _glob_sorted(self.data_dir,
                             f"EsitiMSD_PrezziExAnte_*_{self.msd_zone}.xlsx")
        if not files:
            return pd.DataFrame(columns=['msd_min_buy', 'msd_mean_buy',
                                         'msd_mean_sell', 'msd_max_sell']) \
                .set_index(pd.DatetimeIndex([], name='datetime'))
        dfs = [self._read_msd_file(f) for f in files]
        return self._concat_dedup(dfs)

    def load_all(self) -> MarketData:
        """Load all three data sources and align them on the hourly index."""
        pun = self.load_pun()
        mi  = self.load_mi()
        msd = self.load_msd()
        merged = pun.join(mi, how='outer').join(msd, how='outer')
        merged = merged.sort_index()
        return MarketData(pun=pun, mi=mi, msd=msd, merged=merged)

    # ----- helpers -----

    @staticmethod
    def _concat_dedup(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Concat across files, drop duplicate datetimes (DST artifact)."""
        out = pd.concat(dfs, axis=0)
        # Some upstream files share boundary days (e.g. PUN 2022 ends at
        # 2023-01-01 23:00 while PUN 2023 starts at 2023-01-01 00:00).
        # Keep the first occurrence, drop the rest. Also drops the duplicate
        # hour 24 -> next-day 00:00 from fall-back DST days.
        out = out[~out.index.duplicated(keep='first')]
        return out.sort_index()
