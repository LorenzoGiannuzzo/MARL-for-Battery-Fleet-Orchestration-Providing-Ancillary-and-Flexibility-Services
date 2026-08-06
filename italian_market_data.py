"""
Italian electricity market data layer for the multi-agent BSP pipeline.

Step 8 of the multi-agent track. Provides three things:

  1. PUNSynthetic2024Generator: a parametric generator that produces
     hourly Italian PUN-like price series with statistical properties
     calibrated against the 2024 GME-published series:
       - annual mean ~107 EUR/MWh
       - hourly std ~25 EUR/MWh
       - daily peak at hour 19, trough at hour 04
       - weekend factor ~0.90 (10% lower than weekdays)
       - winter premium / summer discount on baseline
       - log-spike injection for realistic tail behaviour
     Use this for testing the pipeline when actual data is not available.
     The Lorenzo-on-his-side real-data run should use load_pun_from_csv
     pointing to the actual 2024 hourly PUN file from GME.

  2. load_pun_from_csv: a thin loader that accepts a CSV/parquet with
     a 'datetime' column and a 'pun_eur_mwh' column and returns the
     ndarray of hourly prices. Robust to common GME export formats.

  3. build_yearly_service_catalog: builds the FlexibilityService objects
     hour-by-hour for an entire year, applying the time-of-day award
     pattern from Block 7 (night 0.80, peak 0.30) plus mild seasonal
     drift on capacity prices.

The data here is consumed by marl_pipeline.py for multi-day MILP
demonstration generation and out-of-sample evaluation of BC and PPO
policies.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from flexibility_market import FlexibilityService, ServiceType


# ============================================================================
# PUN synthetic generator (calibrated against 2024 statistical properties)
# ============================================================================

@dataclass
class PUN2024Calibration:
    """Statistical parameters of the Italian PUN series in calendar year 2024.

    Sources: GME (Gestore dei Mercati Energetici) public monthly reports;
    consolidated annual review by ARERA and Terna for the bidding-zone-
    weighted national price.
    """
    annual_mean_eur_mwh: float = 107.0
    annual_std_eur_mwh: float = 25.0
    # Diurnal pattern: relative multiplier vs daily mean, by hour-of-day
    # (calibrated so the daily peak hour multiplier ~1.30 and trough ~0.72)
    diurnal_peak_amplitude: float = 0.30
    diurnal_peak_hour: int = 19
    # Weekend price relative to weekday
    weekend_factor: float = 0.90
    # Monthly seasonal multipliers (Jan..Dec), winter higher than summer
    monthly_factors: Tuple[float, ...] = (
        1.18, 1.10, 1.05, 0.95, 0.90, 0.88, 0.92, 0.95, 0.98, 1.02, 1.10, 1.17,
    )
    # AR(1) coefficient for hourly autocorrelation in the noise component
    ar1_coef: float = 0.55
    # Probability per hour of a price spike (multiplier 1.6-3.0x)
    spike_probability: float = 0.003
    # Floor and cap for clipping (no negative or absurd prices)
    floor_eur_mwh: float = 5.0
    cap_eur_mwh:   float = 600.0


class PUNSynthetic2024Generator:
    """Generate hourly PUN-like price series for a contiguous date range.

    Usage:

        gen = PUNSynthetic2024Generator(seed=0)
        prices = gen.generate(start_date=datetime(2024,1,1),
                              end_date=datetime(2024,1,31))
        # prices.shape == (744,)  (31 days * 24 hours)
    """

    def __init__(
        self,
        calibration: Optional[PUN2024Calibration] = None,
        seed: int = 0,
    ):
        self.calibration = calibration or PUN2024Calibration()
        self.rng = np.random.default_rng(seed)

    def _diurnal_factor(self, hour_of_day: int) -> float:
        c = self.calibration
        return 1.0 + c.diurnal_peak_amplitude * math.cos(
            2 * math.pi * (hour_of_day - c.diurnal_peak_hour) / 24.0
        )

    def _weekly_factor(self, weekday: int) -> float:
        # weekday: 0=Monday, 6=Sunday. Weekend = Sat (5), Sun (6).
        return self.calibration.weekend_factor if weekday >= 5 else 1.0

    def _monthly_factor(self, month: int) -> float:
        return self.calibration.monthly_factors[(month - 1) % 12]

    def generate(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> np.ndarray:
        """Generate hourly prices in [start_date, end_date) inclusive of start.

        Returns
        -------
        ndarray of shape (n_hours,) with float64 prices in EUR/MWh.
        """
        c = self.calibration
        n_days = (end_date - start_date).days
        n_hours = n_days * 24
        prices = np.zeros(n_hours)
        # AR(1) noise process around 0
        noise = 0.0
        # Pre-tabulate per-hour multiplicative factors
        for h in range(n_hours):
            dt = start_date + timedelta(hours=h)
            d_factor = self._diurnal_factor(dt.hour)
            w_factor = self._weekly_factor(dt.weekday())
            m_factor = self._monthly_factor(dt.month)
            # Deterministic component
            mean_h = c.annual_mean_eur_mwh * d_factor * w_factor * m_factor
            # Multiplicative AR(1) Gaussian noise; std scaled to target volatility
            innovation = self.rng.normal(0.0, c.annual_std_eur_mwh * 0.6)
            noise = c.ar1_coef * noise + math.sqrt(1 - c.ar1_coef ** 2) * innovation
            value = mean_h + noise
            # Occasional log-spike (rare, multiplicative)
            if self.rng.random() < c.spike_probability:
                value = value * self.rng.uniform(1.6, 3.0)
            value = float(np.clip(value, c.floor_eur_mwh, c.cap_eur_mwh))
            prices[h] = value
        return prices


# ============================================================================
# Real-data loader (CSV)
# ============================================================================

def load_pun_from_csv(
    csv_path: str,
    datetime_col: str = "datetime",
    price_col: str = "pun_eur_mwh",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Tuple[np.ndarray, List[datetime]]:
    """Load hourly PUN prices from a CSV file.

    Accepts standard GME export format with a datetime column and a price
    column (EUR per MWh). Returns the price array and the list of
    datetime stamps for verification.

    The function is intentionally simple and only handles flat files; for
    parquet or excel, swap pd.read_csv with the appropriate reader.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file. The first row is assumed to be the header.
    datetime_col : str
        Name of the datetime column. Expected to parse with pandas default
        datetime parser.
    price_col : str
        Name of the price column, in EUR/MWh.
    start_date, end_date : datetime, optional
        Bound the returned series to this range. If None, return the full
        file content.
    """
    import pandas as pd  # local import to keep top-level light
    df = pd.read_csv(csv_path)
    if datetime_col not in df.columns:
        raise ValueError(f"CSV missing column '{datetime_col}'; got {list(df.columns)}")
    if price_col not in df.columns:
        raise ValueError(f"CSV missing column '{price_col}'; got {list(df.columns)}")
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)
    if start_date is not None:
        df = df[df[datetime_col] >= start_date]
    if end_date is not None:
        df = df[df[datetime_col] < end_date]
    prices = df[price_col].to_numpy(dtype=np.float64)
    stamps = [pd.Timestamp(ts).to_pydatetime() for ts in df[datetime_col]]
    return prices, stamps


def load_pun_from_gme_xlsx(
    xlsx_path: str,
    sheet_name: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    price_column: str = "PUN",
) -> Tuple[np.ndarray, List[datetime]]:
    """Load hourly day-ahead prices from the official GME .xlsx download.

    ZONAL vs PUN (Reviewer 2, point 3-ii)
    -------------------------------------
    The PUN is the national single price paid by CONSUMERS. Generation and
    storage settle at the ZONAL price of the bidding zone they sit in. The
    paper states that MSD data is taken for the northern zone, so the
    day-ahead leg has to be the NORD zonal series, not the PUN, or the two
    revenue streams are drawn from two different price references.

    `price_column` selects which series to read:

        "PUN"   the national single price (legacy behaviour). Correct for a
                consumer-side study, wrong for a storage operator.
        "NORD"  northern-zone price. This is the one a BESS in the north
                actually settles at, and the one to use here.
        "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD" other zones.

    The GME "MGPPrezzi" workbook carries one column per zone plus PUN, so a
    zonal run needs that file rather than the "MGP-PUNPUN" export, which
    only carries the national price. When the requested column is absent the
    function raises rather than silently falling back, because a silent
    fallback to PUN is exactly the failure this parameter exists to prevent.

    The GME download is a single-sheet Excel file (the active sheet is
    typically named 'MGP-PUNPUN' or 'MGP-Prezzi-PUN') with three columns:

        Data   : date in DD/MM/YYYY string format
        Ora    : sequential hour index within the day. By GME convention
                 Ora=1 corresponds to the FIRST clock interval of the day;
                 for normal 24-hour days this is 00:00-01:00 (local time).
        €/MWh  : price as a string with comma decimal separator (Italian
                 locale), e.g. '107,090000'.

    DST handling: in 2024 the spring transition (31 March) has 23 hourly
    entries (Ora 1-23) and the autumn transition (27 October) has 25
    entries (Ora 1-25). To produce a CONTIGUOUS 24-hour-per-day output
    suitable for the MILP/env pipeline:
      - On a short day (23 entries) we DUPLICATE the first value of the
        gap to fill position 2 of the standardised 24-hour vector.
      - On a long day (25 entries) we DROP the extra entry (the DST
        duplicate), keeping the 24 standardised hours of the day.
    These choices are approximations; for a paper-grade run we recommend
    trimming the test window to avoid DST days, or post-processing the
    array externally with the policy of your choice.

    Parameters
    ----------
    xlsx_path : str
        Path to the .xlsx file.
    sheet_name : str, optional
        Name of the sheet to read. If None, the active (first) sheet is
        used, which matches the standard GME export.
    start_date, end_date : datetime, optional
        Bound the returned series. If supplied, only data in
        [start_date, end_date) is returned.

    Returns
    -------
    prices : np.ndarray of shape (n_hours,)
        Hourly PUN prices in EUR/MWh, on a 24-hour-per-day grid.
    stamps : list of datetime, length n_hours
        Hour-start datetime for each price.
    """
    from openpyxl import load_workbook
    from collections import defaultdict

    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    if sheet_name is None:
        ws = wb.active
    else:
        ws = wb[sheet_name]

    # First pass: group entries by date in file order
    rows_by_date: "OrderedDict[date, List[float]]" = {}
    import collections as _collections
    rows_by_date = _collections.OrderedDict()
    seen_header = False
    # Column index of the requested price series, resolved from the header
    # row. The single-price "MGP-PUNPUN" export has the price in column 2;
    # the multi-zone "MGPPrezzi" export has one column per bidding zone, so
    # the index has to be looked up by name.
    price_idx = 2
    want = str(price_column).strip().upper()
    for row in ws.iter_rows(values_only=True):
        if row is None or row[0] is None:
            continue
        if not seen_header:
            seen_header = True
            header = [str(c).strip().upper() if c is not None else ""
                      for c in row]
            matches = [i for i, h in enumerate(header) if h == want]
            if matches:
                price_idx = matches[0]
            elif want not in ("PUN", "\u20ac/MWH", "EUR/MWH"):
                raise ValueError(
                    f"price_column={price_column!r} not found in {xlsx_path}. "
                    f"Header columns: {header}. The single-price "
                    f"'MGP-PUNPUN' export does not contain zonal columns; "
                    f"use the multi-zone 'MGPPrezzi' workbook for zonal "
                    f"prices.")
            continue
        data_str, ora, price = row[0], row[1], row[price_idx]
        if isinstance(data_str, datetime):
            day = data_str.date()
        else:
            day = datetime.strptime(str(data_str), "%d/%m/%Y").date()
        if isinstance(price, str):
            price_f = float(price.replace(",", "."))
        else:
            price_f = float(price)
        rows_by_date.setdefault(day, []).append(price_f)

    # Second pass: build the 24-hour-per-day output, handling DST anomalies
    prices_list: List[float] = []
    stamps_list: List[datetime] = []
    for day, entries in rows_by_date.items():
        n = len(entries)
        if n == 24:
            normalised = list(entries)
        elif n == 23:
            # Spring DST: insert a synthetic value at position 2 (the missing
            # 02:00-03:00 local hour). Use the average of its neighbours.
            normalised = list(entries[:2]) + \
                          [(entries[1] + entries[2]) / 2.0] + \
                          list(entries[2:])
        elif n == 25:
            # Autumn DST: drop the duplicate hour (the second 02:00-03:00).
            # GME puts the duplicate at Ora=4 in this file, but for safety
            # we drop position 3 and keep position 2.
            normalised = list(entries[:3]) + list(entries[4:])
        else:
            # Unexpected day length; truncate or pad to 24
            if n > 24:
                normalised = list(entries[:24])
            else:
                normalised = list(entries) + [entries[-1]] * (24 - n)
        for h in range(24):
            ts = datetime(day.year, day.month, day.day, h)
            if start_date is not None and ts < start_date:
                continue
            if end_date is not None and ts >= end_date:
                continue
            prices_list.append(normalised[h])
            stamps_list.append(ts)

    prices = np.asarray(prices_list, dtype=np.float64)
    return prices, stamps_list


def load_pun_from_gme_xlsx_multi(
    xlsx_paths,
    sheet_name: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    price_column: str = "PUN",
) -> Tuple[np.ndarray, List[datetime]]:
    """Load and concatenate hourly PUN prices from one or more GME .xlsx files.

    Accepts either a single path (str) or a list of paths. When a list is
    given, each file is loaded with ``load_pun_from_gme_xlsx``, the resulting
    series are concatenated, **deduplicated by timestamp** (last wins, so
    later files in the list override earlier ones for overlapping hours), and
    sorted chronologically. The optional ``start_date``/``end_date`` window
    is applied AFTER merging, so it works regardless of how the files are
    split.

    Use this when train and test windows span calendar years and you have
    one xlsx per year (e.g. ``["2023_PUN.xlsx", "2024_PUN.xlsx"]``).
    """
    if isinstance(xlsx_paths, (str, bytes)):
        xlsx_paths = [xlsx_paths]

    all_prices: List[float] = []
    all_stamps: List[datetime] = []
    for p in xlsx_paths:
        prices_i, stamps_i = load_pun_from_gme_xlsx(
            p, sheet_name=sheet_name, price_column=price_column)
        all_prices.extend(prices_i.tolist())
        all_stamps.extend(stamps_i)

    # Dedup by timestamp (last write wins) and sort.
    by_ts: "OrderedDict[datetime, float]" = OrderedDict()
    for ts, pr in zip(all_stamps, all_prices):
        by_ts[ts] = float(pr)
    sorted_items = sorted(by_ts.items(), key=lambda kv: kv[0])
    stamps_sorted = [ts for ts, _ in sorted_items]
    prices_sorted = np.array([pr for _, pr in sorted_items], dtype=float)

    # Window filter (after merge).
    if start_date is not None or end_date is not None:
        mask = np.ones(len(stamps_sorted), dtype=bool)
        if start_date is not None:
            mask &= np.array([ts >= start_date for ts in stamps_sorted])
        if end_date is not None:
            mask &= np.array([ts < end_date for ts in stamps_sorted])
        prices_sorted = prices_sorted[mask]
        stamps_sorted = [ts for ts, m in zip(stamps_sorted, mask) if m]

    return prices_sorted, stamps_sorted


def make_market_window_from_real_pun(
    start_date: datetime,
    end_date: datetime,
    pun_xlsx_path,  # str or List[str]
    # Day-ahead price series. "PUN" is the national consumer reference
    # price; a storage operator settles at the ZONAL price, so a study of
    # a northern-zone fleet whose MSD data is EsitiMSD_*_Nord must set
    # this to "NORD" (Reviewer 2, point 3-ii).
    price_column: str = "PUN",
    service_calibration: Optional[ServiceCalibration] = None,
    msd_xlsx_paths: Optional[List[str]] = None,
    msd_use_empirical_award_rates: bool = True,
    msd_year_for_profiles: int = 2024,
    directional_services: bool = False,
    msd_conditioned: bool = True,
) -> "MarketWindow":
    """Build a MarketWindow using real GME PUN prices and (optionally) real
    Terna MSD ex-ante data for both energy prices and empirical award rates.

    PUN drives the day-ahead arbitrage component. When `msd_xlsx_paths` is
    supplied, MSD data feeds the per-hour energy_price and (if
    `msd_use_empirical_award_rates=True`) the per-hour award_probability of
    aFRR/mFRR. The mapping is:
      - aFRR <- upward MSD (Prezzo Medio di Acquisto): activation rate and
        mean price-when-active, both as 24-value per-hour-of-day profiles
        derived from the full year. Tiled across the date range.
      - mFRR <- downward MSD (Prezzo Medio di Vendita) with a 0.6 multiplier
        on award rate to reflect the manual-vs-automatic nature.
      - FCR  unchanged, kept synthetic from ServiceCalibration (FCR is
        cleared on a separate European market, not in MSD).
    Capacity prices and activation_probability remain at ServiceCalibration
    defaults for all three services.

    Parameters
    ----------
    start_date, end_date : datetime
    pun_xlsx_path : str
    service_calibration : ServiceCalibration, optional
    msd_xlsx_paths : list of str, optional
    msd_use_empirical_award_rates : bool
        When True (default), the empirical activation rate per hour-of-day
        from the MSD year (default 2024) replaces the synthetic
        time-of-day award_probability profile.
    msd_year_for_profiles : int
        Calendar year used to build the 24-value hourly profiles.
    """
    prices, stamps = load_pun_from_gme_xlsx_multi(
        pun_xlsx_path, start_date=start_date, end_date=end_date,
        price_column=price_column,
    )
    expected_hours = int((end_date - start_date).total_seconds() // 3600)
    if len(prices) != expected_hours:
        if len(prices) < expected_hours:
            pad = np.full(expected_hours - len(prices),
                          float(prices[-1]) if len(prices) else 0.0)
            prices = np.concatenate([prices, pad])
        else:
            prices = prices[:expected_hours]

    overrides: Dict[str, Optional[np.ndarray]] = {
        "afrr_energy_overrides": None, "mfrr_energy_overrides": None,
        "afrr_award_overrides":  None, "mfrr_award_overrides":  None,
    }
    if msd_xlsx_paths:
        if msd_use_empirical_award_rates:
            if msd_conditioned:
                # Option 2: profiles conditioned on (season x daytype x hour),
                # using both 2023 and 2024 (year=None) for ~2x samples per cell.
                # Day-by-day variability in award rates and active prices.
                cond = msd_conditioned_profiles(msd_xlsx_paths, year=None)
                overrides.update(build_overrides_from_conditioned_profiles(
                    cond, start_date, end_date,
                    directional=directional_services,
                ))
                mode = "directional 5-service" if directional_services else "legacy 3-service"
                print(f"  MSD profiles (CONDITIONED season x daytype x hour, "
                      f"{mode}): {len(cond['buckets'])} buckets, hourly fallback "
                      f"on sparse cells; upward award fb-range "
                      f"[{cond['fallback_upward_award_rate'].min():.2f}, "
                      f"{cond['fallback_upward_award_rate'].max():.2f}], "
                      f"upward price-when-active fb-range "
                      f"[{cond['fallback_upward_price_when_active'].min():.0f}, "
                      f"{cond['fallback_upward_price_when_active'].max():.0f}] EUR/MWh")
            else:
                # Real per-hour-of-day profiles from the MSD year (legacy)
                profiles = msd_hourly_profiles(msd_xlsx_paths,
                                                year=msd_year_for_profiles)
                overrides.update(build_overrides_from_msd_profiles(
                    profiles, start_date, end_date,
                    directional=directional_services,
                ))
                mode = "directional 5-service" if directional_services else "legacy 3-service"
                print(f"  MSD profiles ({msd_year_for_profiles}, {mode}): "
                      f"upward award rate hour-of-day [{profiles['upward_award_rate'].min():.2f}, "
                      f"{profiles['upward_award_rate'].max():.2f}], "
                      f"upward price-when-active [{profiles['upward_price_when_active'].min():.0f}, "
                      f"{profiles['upward_price_when_active'].max():.0f}] EUR/MWh; "
                      f"downward [{profiles['downward_award_rate'].min():.2f}, "
                      f"{profiles['downward_award_rate'].max():.2f}], "
                      f"[{profiles['downward_price_when_active'].min():.0f}, "
                      f"{profiles['downward_price_when_active'].max():.0f}]")
        else:
            # Legacy behaviour: tile MSD per-hour series as energy-only
            msd_hourly = load_msd_from_terna_xlsx(
                msd_xlsx_paths, start_date=start_date, end_date=end_date,
                use_field="prezzo_medio_vendita",
            )
            overrides["afrr_energy_overrides"] = msd_hourly
            overrides["mfrr_energy_overrides"] = msd_hourly

    services = build_yearly_service_catalog(
        start_date, end_date, service_calibration,
        **overrides,
    )
    return MarketWindow(
        start_date=start_date,
        end_date=end_date,
        prices_hourly=prices,
        services_hourly=services,
    )


# ============================================================================
# Annual flexibility service catalog
# ============================================================================

@dataclass
class ServiceCalibration:
    """Italian ancillary service capacity/energy prices.

    Defaults are CALIBRATED against published 2022-2024 Italian benchmarks
    documented in the user's Italian_BESS_Markets_Reference workbook
    (build 2026-05-22, version 1.0). Source attributions:

      - Capacity prices: Magnus Energy 2025 "Developments in balancing and
        capacity markets"; ACER CHEST database; Terna Report on Balancing
        2022-2023. FCR uses the European FCR Cooperation 2024 benchmark
        because Italian market-based FCR is not yet operational under TIDE.
      - Energy activation prices: ACER CHEST 2022 volume-weighted aFRR
        Italy upward (411.99 EUR/MWh, rounded to 250 conservative);
        mFRR European benchmark.
      - Activation probabilities: Terna Report on Balancing 2022-2023.

    For real hourly variation in energy prices, supply the per-hour
    `energy_price_overrides` arrays (FCR / aFRR / mFRR) to
    `build_yearly_service_catalog`. Typically these come from the MSD
    ex-ante data via `load_msd_from_terna_xlsx`.
    """
    # Capacity prices (EUR per MW per hour) — calibrated to 2022-2024 benchmarks
    fcr_capacity_price: float = 50.0       # European FCR Cooperation 2024 avg
    afrr_capacity_price: float = 25.0      # Italian aFRR benchmark post-PICASSO
    mfrr_capacity_price: float = 12.0      # European mFRR/MARI benchmark
    # Energy prices for activated capacity (EUR per MWh) — calibrated
    fcr_energy_price: float = 100.0        # bundled in capacity in most EU mkts
    afrr_energy_price: float = 250.0       # Italian 2022 vol-weighted ~412; conservative
    mfrr_energy_price: float = 180.0       # European mFRR 2022-2024 benchmark
    # Baseline activation probabilities — Terna Report on Balancing 2022-23
    fcr_activation_probability: float = 0.10
    afrr_activation_probability: float = 0.30
    mfrr_activation_probability: float = 0.20
    # Time-of-day award probability pattern (Block 7 single-BESS recipe)
    award_prob_night: float = 0.80      # 00-06
    award_prob_morning: float = 0.55    # 06-09, 09-12, 12-17
    award_prob_peak: float = 0.30       # 17-21
    award_prob_evening: float = 0.50    # 21-24
    # Service-specific offsets vs the time-of-day baseline
    fcr_award_offset:  float = -0.15
    afrr_award_offset: float =  0.10
    mfrr_award_offset: float =  0.00


def _time_of_day_award_probability(hour: int, base_calibration: ServiceCalibration) -> float:
    """Return the baseline award probability for the given hour of day."""
    if 0 <= hour < 6:
        return base_calibration.award_prob_night
    elif 6 <= hour < 17:
        return base_calibration.award_prob_morning
    elif 17 <= hour < 21:
        return base_calibration.award_prob_peak
    else:
        return base_calibration.award_prob_evening


def load_msd_from_terna_xlsx(
    xlsx_paths,
    start_date: datetime,
    end_date: datetime,
    sheet_name: Optional[str] = None,
    use_field: str = "prezzo_medio_vendita",
) -> np.ndarray:
    """Load hourly MSD ex-ante prices from one or more Terna .xlsx exports.

    The Terna `EsitiMSD_PrezziExAnte_*_Nord.xlsx` files have a single sheet
    `PrezziExAnte_NORD_MSD` with columns:

        Data
        Ora                                 (1-24 in GME convention)
        Prezzo Minimo di Acquisto (EUR/MWh)   -> use_field='prezzo_minimo_acquisto'
        Prezzo Medio di Acquisto (EUR/MWh)    -> use_field='prezzo_medio_acquisto'
        Prezzo Medio di Vendita (EUR/MWh)     -> use_field='prezzo_medio_vendita' (default)
        Prezzo Massimo di Vendita (EUR/MWh)   -> use_field='prezzo_massimo_vendita'

    For BESS modelling, the most useful proxy of the per-hour activation
    energy revenue is `prezzo_medio_vendita`: the average price at which
    Terna procured upward regulation in MSD ex-ante (i.e., the price a
    BESS gets paid per MWh injected on upward activation). Values of 0
    indicate no upward activation in that hour; for the service catalog
    we keep the calibrated baseline `energy_price` from
    `ServiceCalibration` in those cases (a zero would zero out the
    expected revenue, which would not reflect that the BESS still gets
    capacity payment).

    Multiple input files are concatenated and de-duplicated by timestamp
    (the Terna files cover overlapping windows in some cases).

    Parameters
    ----------
    xlsx_paths : str or list of str
        Path(s) to one or more Terna MSD .xlsx files.
    start_date, end_date : datetime
        Bounding the returned array. The output has exactly
        (end_date - start_date) hours of data.
    sheet_name : str, optional
        Override the sheet name; default uses the active sheet.
    use_field : str
        Which Terna column to extract. Default 'prezzo_medio_vendita'.

    Returns
    -------
    np.ndarray of shape (n_hours,) with float prices in EUR/MWh. Zeros
    where Terna reported no activation; NaN where data is missing entirely.
    """
    from openpyxl import load_workbook
    if isinstance(xlsx_paths, str):
        xlsx_paths = [xlsx_paths]

    field_to_col = {
        "prezzo_minimo_acquisto": 2,
        "prezzo_medio_acquisto":  3,
        "prezzo_medio_vendita":   4,
        "prezzo_massimo_vendita": 5,
    }
    if use_field not in field_to_col:
        raise ValueError(f"use_field must be one of {list(field_to_col)}")
    col_idx = field_to_col[use_field]

    by_ts: Dict[datetime, float] = {}
    for path in xlsx_paths:
        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb[sheet_name] if sheet_name else wb.active
        seen_header = False
        for row in ws.iter_rows(values_only=True):
            if row is None or row[0] is None:
                continue
            if not seen_header:
                seen_header = True
                continue
            data_str, ora = row[0], row[1]
            value = row[col_idx]
            if isinstance(data_str, datetime):
                day = data_str.date()
            else:
                day = datetime.strptime(str(data_str), "%d/%m/%Y").date()
            # Hour code: 1-24 -> 0-23. We don't try to handle DST anomalies for
            # MSD; in our pipeline the MSD prices are an OVERLAY on the
            # PUN-derived 24-hour grid, and DST off-by-ones in MSD shift by
            # one hour on transition days, which is acceptable for capacity
            # market modelling.
            try:
                hour = int(ora) - 1
                if hour < 0 or hour > 23:
                    continue
            except (TypeError, ValueError):
                continue
            ts = datetime(day.year, day.month, day.day, hour)
            if isinstance(value, str):
                value_f = float(value.replace(".", "").replace(",", "."))
            else:
                value_f = float(value) if value is not None else 0.0
            by_ts[ts] = value_f
        wb.close()

    # Materialise to a contiguous array over [start_date, end_date)
    n_hours = int((end_date - start_date).total_seconds() // 3600)
    out = np.full(n_hours, np.nan, dtype=np.float64)
    for i in range(n_hours):
        ts = start_date + timedelta(hours=i)
        if ts in by_ts:
            out[i] = by_ts[ts]
    return out


def msd_hourly_profiles(
    xlsx_paths,
    year: int = 2024,
) -> Dict[str, np.ndarray]:
    """Extract per-hour-of-day MSD ex-ante profiles for a given year.

    Reads one or more Terna `EsitiMSD_PrezziExAnte_*_Nord.xlsx` files,
    filters to the chosen year, and aggregates per hour-of-day (0..23) the
    following four arrays of length 24:

      - `upward_award_rate`: fraction of hours-of-day where the upward
        clearing price (Prezzo Medio di Acquisto) is > 0. Proxy for the
        empirical activation rate of upward reserves at that hour.
      - `downward_award_rate`: same for downward (Prezzo Medio di Vendita > 0).
      - `upward_price_when_active`: mean clearing price across the year for
        the hours-of-day when active (EUR/MWh). Proxy for energy_price
        received by a BSP discharging on upward activation.
      - `downward_price_when_active`: same for downward.

    These profiles are then used as `award_probability` and `energy_price`
    overrides in `build_yearly_service_catalog`, replacing the synthetic
    time-of-day pattern with empirical Italian 2024 ones. Mapping to the
    three services in our model is approximate (the MSD ex-ante does not
    distinguish FCR vs aFRR vs mFRR explicitly):

      - aFRR <- upward_*  (MSD ex-ante is the main automatic upward market)
      - mFRR <- downward_* with a 0.6 multiplier on award_rate (mFRR is
                manual and less frequent than aFRR)
      - FCR  unchanged, kept synthetic with ServiceCalibration (FCR is
             cleared on a separate European market, not present in MSD)

    Parameters
    ----------
    xlsx_paths : str or list of str
    year : int
        Filter to rows of this calendar year. Defaults to 2024.

    Returns
    -------
    dict with the four arrays above, keys 'upward_award_rate',
    'downward_award_rate', 'upward_price_when_active',
    'downward_price_when_active'.
    """
    import pandas as pd
    if isinstance(xlsx_paths, (str, bytes)):
        xlsx_paths = [xlsx_paths]
    dfs = []
    for p in xlsx_paths:
        df = pd.read_excel(p)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    def to_float(s):
        return pd.to_numeric(s.astype(str).str.replace(',', '.'),
                              errors='coerce')

    df['date'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
    df['hour'] = df['Ora'].astype(int) - 1  # Terna 1-24 -> 0-23
    df['p_buy_mean']  = to_float(df['Prezzo Medio di Acquisto (€/MWh)'])
    df['p_sell_mean'] = to_float(df['Prezzo Medio di Vendita (€/MWh)'])

    df = df[df['date'].dt.year == year]
    df = df[(df['hour'] >= 0) & (df['hour'] <= 23)]

    up_rate   = np.zeros(24, dtype=np.float64)
    up_price  = np.zeros(24, dtype=np.float64)
    dn_rate   = np.zeros(24, dtype=np.float64)
    dn_price  = np.zeros(24, dtype=np.float64)
    for h in range(24):
        sub = df[df['hour'] == h]
        if len(sub) == 0:
            continue
        up_rate[h]  = float((sub['p_buy_mean'] > 0).mean())
        active_up = sub[sub['p_buy_mean'] > 0]
        up_price[h] = float(active_up['p_buy_mean'].mean()) if len(active_up) > 0 else 0.0
        dn_rate[h]  = float((sub['p_sell_mean'] > 0).mean())
        active_dn = sub[sub['p_sell_mean'] > 0]
        dn_price[h] = float(active_dn['p_sell_mean'].mean()) if len(active_dn) > 0 else 0.0
    return {
        "upward_award_rate": up_rate,
        "upward_price_when_active": up_price,
        "downward_award_rate": dn_rate,
        "downward_price_when_active": dn_price,
    }


def build_overrides_from_msd_profiles(
    profiles: Dict[str, np.ndarray],
    start_date: datetime,
    end_date: datetime,
    mfrr_award_scale: float = 0.6,
    directional: bool = False,
) -> Dict[str, np.ndarray]:
    """Tile 24-hour MSD profiles into per-hour override arrays for the
    full date range, ready to feed into build_yearly_service_catalog.

    Two modes:

    Legacy (`directional=False`, default for back-compat):
      - aFRR award_probability  = upward_award_rate[hour_of_day]
      - aFRR energy_price       = upward_price_when_active[hour_of_day]
      - mFRR award_probability  = downward_award_rate[hour_of_day] * mfrr_award_scale
      - mFRR energy_price       = downward_price_when_active[hour_of_day]
      Single direction aggregated, used by 3-service catalog.

    Directional (`directional=True`):
      - aFRR_up award/energy  <- upward MSD (Prezzo Medio di Acquisto)
      - aFRR_dn award/energy  <- downward MSD (Prezzo Medio di Vendita)
      - mFRR_up award/energy  <- upward MSD with award scaled by mfrr_award_scale
      - mFRR_dn award/energy  <- downward MSD with award scaled by mfrr_award_scale
      Used by 5-service directional catalog.

    FCR is never in MSD; keep synthetic via ServiceCalibration.
    """
    n_days = (end_date - start_date).days
    n_hours = n_days * 24
    up_rate  = profiles["upward_award_rate"]
    up_price = profiles["upward_price_when_active"]
    dn_rate  = profiles["downward_award_rate"]
    dn_price = profiles["downward_price_when_active"]

    if not directional:
        afrr_award = np.zeros(n_hours, dtype=np.float64)
        afrr_energy = np.zeros(n_hours, dtype=np.float64)
        mfrr_award = np.zeros(n_hours, dtype=np.float64)
        mfrr_energy = np.zeros(n_hours, dtype=np.float64)
        for h in range(n_hours):
            dt = start_date + timedelta(hours=h)
            hh = dt.hour
            afrr_award[h]  = up_rate[hh]
            afrr_energy[h] = up_price[hh]
            mfrr_award[h]  = dn_rate[hh] * mfrr_award_scale
            mfrr_energy[h] = dn_price[hh]
        return {
            "afrr_award_overrides":  afrr_award,
            "afrr_energy_overrides": afrr_energy,
            "mfrr_award_overrides":  mfrr_award,
            "mfrr_energy_overrides": mfrr_energy,
        }

    # Directional mode: emit overrides for both directions of both services.
    # IMPORTANT: empirical upward and downward award rates are mutex-normalised
    # so that for any hour P_up_excl + P_dn_excl <= 1, representing the
    # categorical sample on direction Terna calls. When sum > 1 in the raw
    # data (typical: hour 18-21 where both directions are active in different
    # quarters of the hour), we normalise proportionally with P_none=0. This
    # reflects the operational reality that for any settlement period Terna
    # calls EITHER upward OR downward, not both simultaneously.
    afrr_up_award = np.zeros(n_hours, dtype=np.float64)
    afrr_up_energy = np.zeros(n_hours, dtype=np.float64)
    afrr_dn_award = np.zeros(n_hours, dtype=np.float64)
    afrr_dn_energy = np.zeros(n_hours, dtype=np.float64)
    mfrr_up_award = np.zeros(n_hours, dtype=np.float64)
    mfrr_up_energy = np.zeros(n_hours, dtype=np.float64)
    mfrr_dn_award = np.zeros(n_hours, dtype=np.float64)
    mfrr_dn_energy = np.zeros(n_hours, dtype=np.float64)
    for h in range(n_hours):
        dt = start_date + timedelta(hours=h)
        hh = dt.hour
        # Mutex-aware probabilities
        p_up_raw = up_rate[hh]
        p_dn_raw = dn_rate[hh]
        total = p_up_raw + p_dn_raw
        if total <= 1.0:
            p_up_excl = p_up_raw
            p_dn_excl = p_dn_raw
        else:
            # Normalise so they sum to 1 (P_none=0): hours where Terna
            # always calls something, split proportionally.
            p_up_excl = p_up_raw / total
            p_dn_excl = p_dn_raw / total
        afrr_up_award[h]  = p_up_excl
        afrr_up_energy[h] = up_price[hh]
        afrr_dn_award[h]  = p_dn_excl
        afrr_dn_energy[h] = dn_price[hh]
        mfrr_up_award[h]  = p_up_excl * mfrr_award_scale
        mfrr_up_energy[h] = up_price[hh]
        mfrr_dn_award[h]  = p_dn_excl * mfrr_award_scale
        mfrr_dn_energy[h] = dn_price[hh]
    return {
        "afrr_up_award_overrides":  afrr_up_award,
        "afrr_up_energy_overrides": afrr_up_energy,
        "afrr_dn_award_overrides":  afrr_dn_award,
        "afrr_dn_energy_overrides": afrr_dn_energy,
        "mfrr_up_award_overrides":  mfrr_up_award,
        "mfrr_up_energy_overrides": mfrr_up_energy,
        "mfrr_dn_award_overrides":  mfrr_dn_award,
        "mfrr_dn_energy_overrides": mfrr_dn_energy,
    }


def build_yearly_service_catalog(
    start_date: datetime,
    end_date: datetime,
    calibration: Optional[ServiceCalibration] = None,
    afrr_energy_overrides: Optional[np.ndarray] = None,
    mfrr_energy_overrides: Optional[np.ndarray] = None,
    fcr_energy_overrides: Optional[np.ndarray] = None,
    afrr_award_overrides: Optional[np.ndarray] = None,
    mfrr_award_overrides: Optional[np.ndarray] = None,
    fcr_award_overrides: Optional[np.ndarray] = None,
    # NEW directional inputs - if any are provided, builder emits the 5-service
    # directional catalog (FCR + aFRR_up + aFRR_dn + mFRR_up + mFRR_dn)
    afrr_up_energy_overrides: Optional[np.ndarray] = None,
    afrr_dn_energy_overrides: Optional[np.ndarray] = None,
    mfrr_up_energy_overrides: Optional[np.ndarray] = None,
    mfrr_dn_energy_overrides: Optional[np.ndarray] = None,
    afrr_up_award_overrides: Optional[np.ndarray] = None,
    afrr_dn_award_overrides: Optional[np.ndarray] = None,
    mfrr_up_award_overrides: Optional[np.ndarray] = None,
    mfrr_dn_award_overrides: Optional[np.ndarray] = None,
) -> List[List[FlexibilityService]]:
    """Build the per-hour FlexibilityService catalog for a contiguous date range.

    Returns a list of length n_hours, each entry a list of 3 services
    (FCR, aFRR, mFRR). The award_probability of each service combines the
    time-of-day base pattern with the service-specific offset, then clips
    to [0.05, 0.95].

    Optional per-hour overrides (priority over the synthetic profile when
    set, else fall back to calibration default):
      - `afrr_energy_overrides`, `mfrr_energy_overrides`, `fcr_energy_overrides`:
        replace the calibration's energy_price for the relevant service.
      - `afrr_award_overrides`, `mfrr_award_overrides`, `fcr_award_overrides`:
        replace the synthetic time-of-day award probability with a real
        empirical value (e.g. MSD activation rate per hour). Values of NaN
        or <=0 fall back to the synthetic profile for that hour.
    All overrides have shape (n_hours,) aligned with the catalog.
    """
    cal = calibration or ServiceCalibration()
    n_days = (end_date - start_date).days
    n_hours = n_days * 24
    catalog: List[List[FlexibilityService]] = []

    def _override(arr, default, h):
        if arr is None:
            return default
        if h >= len(arr):
            return default
        v = arr[h]
        if v is None or (isinstance(v, float) and (np.isnan(v) or v <= 0.0)):
            return default
        return float(v)

    # Directional mode triggered if ANY directional override is provided
    directional_mode = any(arr is not None for arr in (
        afrr_up_energy_overrides, afrr_dn_energy_overrides,
        mfrr_up_energy_overrides, mfrr_dn_energy_overrides,
        afrr_up_award_overrides,  afrr_dn_award_overrides,
        mfrr_up_award_overrides,  mfrr_dn_award_overrides,
    ))

    for h in range(n_hours):
        dt = start_date + timedelta(hours=h)
        base = _time_of_day_award_probability(dt.hour, cal)
        season_offset = 0.05 if dt.month in (1, 2, 12) else 0.0
        fcr_p_synth  = float(np.clip(base + cal.fcr_award_offset  - season_offset, 0.05, 0.95))
        afrr_p_synth = float(np.clip(base + cal.afrr_award_offset - season_offset, 0.05, 0.95))
        mfrr_p_synth = float(np.clip(base + cal.mfrr_award_offset - season_offset, 0.05, 0.95))
        fcr_p  = _override(fcr_award_overrides,  fcr_p_synth,  h)
        fcr_p  = float(np.clip(fcr_p,  0.05, 0.99))
        fcr_e  = _override(fcr_energy_overrides, cal.fcr_energy_price, h)

        if directional_mode:
            # Build the 5-service directional catalog
            afrr_up_p = float(np.clip(_override(afrr_up_award_overrides, afrr_p_synth, h), 0.05, 0.99))
            afrr_dn_p = float(np.clip(_override(afrr_dn_award_overrides, afrr_p_synth, h), 0.05, 0.99))
            mfrr_up_p = float(np.clip(_override(mfrr_up_award_overrides, mfrr_p_synth, h), 0.05, 0.99))
            mfrr_dn_p = float(np.clip(_override(mfrr_dn_award_overrides, mfrr_p_synth, h), 0.05, 0.99))
            afrr_up_e = _override(afrr_up_energy_overrides, cal.afrr_energy_price, h)
            afrr_dn_e = _override(afrr_dn_energy_overrides, cal.afrr_energy_price, h)
            mfrr_up_e = _override(mfrr_up_energy_overrides, cal.mfrr_energy_price, h)
            mfrr_dn_e = _override(mfrr_dn_energy_overrides, cal.mfrr_energy_price, h)
            hour_services = [
                FlexibilityService(
                    service_type=ServiceType.FCR,
                    capacity_price=cal.fcr_capacity_price,
                    energy_price=fcr_e,
                    activation_probability=cal.fcr_activation_probability,
                    response_time=30, min_capacity=1.0,
                    max_duration=4, min_duration=1,
                    award_probability=fcr_p,
                ),
                FlexibilityService(
                    service_type=ServiceType.AFRR_UP,
                    capacity_price=cal.afrr_capacity_price,
                    energy_price=afrr_up_e,
                    activation_probability=cal.afrr_activation_probability,
                    response_time=200, min_capacity=1.0,
                    max_duration=4, min_duration=1,
                    award_probability=afrr_up_p,
                ),
                FlexibilityService(
                    service_type=ServiceType.AFRR_DN,
                    capacity_price=cal.afrr_capacity_price,
                    energy_price=afrr_dn_e,
                    activation_probability=cal.afrr_activation_probability,
                    response_time=200, min_capacity=1.0,
                    max_duration=4, min_duration=1,
                    award_probability=afrr_dn_p,
                ),
                FlexibilityService(
                    service_type=ServiceType.MFRR_UP,
                    capacity_price=cal.mfrr_capacity_price,
                    energy_price=mfrr_up_e,
                    activation_probability=cal.mfrr_activation_probability,
                    response_time=900, min_capacity=1.0,
                    max_duration=4, min_duration=1,
                    award_probability=mfrr_up_p,
                ),
                FlexibilityService(
                    service_type=ServiceType.MFRR_DN,
                    capacity_price=cal.mfrr_capacity_price,
                    energy_price=mfrr_dn_e,
                    activation_probability=cal.mfrr_activation_probability,
                    response_time=900, min_capacity=1.0,
                    max_duration=4, min_duration=1,
                    award_probability=mfrr_dn_p,
                ),
            ]
            catalog.append(hour_services)
            continue

        # Legacy 3-service path (back-compat)
        afrr_p = _override(afrr_award_overrides, afrr_p_synth, h)
        mfrr_p = _override(mfrr_award_overrides, mfrr_p_synth, h)
        afrr_p = float(np.clip(afrr_p, 0.05, 0.99))
        mfrr_p = float(np.clip(mfrr_p, 0.05, 0.99))
        afrr_e = _override(afrr_energy_overrides, cal.afrr_energy_price, h)
        mfrr_e = _override(mfrr_energy_overrides, cal.mfrr_energy_price, h)
        hour_services = [
            FlexibilityService(
                service_type=ServiceType.FCR,
                capacity_price=cal.fcr_capacity_price,
                energy_price=fcr_e,
                activation_probability=cal.fcr_activation_probability,
                response_time=30, min_capacity=1.0,
                max_duration=4, min_duration=1,
                award_probability=fcr_p,
            ),
            FlexibilityService(
                service_type=ServiceType.AFRR,
                capacity_price=cal.afrr_capacity_price,
                energy_price=afrr_e,
                activation_probability=cal.afrr_activation_probability,
                response_time=200, min_capacity=1.0,
                max_duration=4, min_duration=1,
                award_probability=afrr_p,
            ),
            FlexibilityService(
                service_type=ServiceType.MFRR,
                capacity_price=cal.mfrr_capacity_price,
                energy_price=mfrr_e,
                activation_probability=cal.mfrr_activation_probability,
                response_time=900, min_capacity=1.0,
                max_duration=4, min_duration=1,
                award_probability=mfrr_p,
            ),
        ]
        catalog.append(hour_services)
    return catalog


# ============================================================================
# Market window dataclass
# ============================================================================

@dataclass
class MarketWindow:
    """A contiguous period of market data ready to be consumed by the MILP/env."""
    start_date: datetime
    end_date: datetime
    prices_hourly: np.ndarray  # (n_hours,) EUR/MWh
    services_hourly: List[List[FlexibilityService]]  # (n_hours, 3 services)

    @property
    def n_hours(self) -> int:
        return len(self.prices_hourly)

    @property
    def n_days(self) -> int:
        return self.n_hours // 24

    def slice_day(self, day_index: int) -> Tuple[List[float], List[List[FlexibilityService]]]:
        """Return prices and services for day `day_index` (0-based)."""
        h0 = day_index * 24
        h1 = h0 + 24
        return list(self.prices_hourly[h0:h1]), list(self.services_hourly[h0:h1])


def make_synthetic_market_window(
    start_date: datetime,
    end_date: datetime,
    pun_calibration: Optional[PUN2024Calibration] = None,
    service_calibration: Optional[ServiceCalibration] = None,
    seed: int = 0,
) -> MarketWindow:
    """Convenience constructor for testing: generate prices + services together."""
    gen = PUNSynthetic2024Generator(calibration=pun_calibration, seed=seed)
    prices = gen.generate(start_date, end_date)
    services = build_yearly_service_catalog(start_date, end_date, service_calibration)
    return MarketWindow(
        start_date=start_date,
        end_date=end_date,
        prices_hourly=prices,
        services_hourly=services,
    )

# ============================================================================
# OPZIONE 2: profili MSD condizionati su (stagione x tipo-giorno x ora)
# Aggiunto per introdurre variabilità giorno-per-giorno (stagionale + feriale/
# weekend) negli award rate e soprattutto nei prezzi-quando-attivi, che dai
# dati Terna 2023-2024 variano ~20-30% tra inverno ed estate. Retro-compatibile:
# le funzioni esistenti restano intatte; per usare l'Opzione 2 si chiama
# msd_conditioned_profiles + build_overrides_from_conditioned_profiles.
# ============================================================================

_SEASON_MAP_OPT2 = {12:'W',1:'W',2:'W',3:'Sp',4:'Sp',5:'Sp',
                    6:'Su',7:'Su',8:'Su',9:'A',10:'A',11:'A'}


def msd_conditioned_profiles(xlsx_paths, year=None, min_cell_samples: int = 20):
    """MSD ex-ante profiles conditioned on (season x daytype x hour).

    Returns dict with:
      - 'buckets': {(season, daytype): {4 length-24 arrays}}
      - 'fallback_*': hour-only marginal arrays (length 24), used where a cell
        has fewer than `min_cell_samples` observations.
    season in {'W','Sp','Su','A'}; daytype in {'wd','we'}.
    """
    import pandas as pd
    if isinstance(xlsx_paths, (str, bytes)):
        xlsx_paths = [xlsx_paths]
    df = pd.concat([pd.read_excel(p) for p in xlsx_paths], ignore_index=True)

    def _tf(s):
        return pd.to_numeric(s.astype(str).str.replace(',', '.'), errors='coerce')

    df['date'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
    df['hour'] = df['Ora'].astype(int) - 1
    df['p_buy'] = _tf(df['Prezzo Medio di Acquisto (€/MWh)'])
    df['p_sell'] = _tf(df['Prezzo Medio di Vendita (€/MWh)'])
    df = df.dropna(subset=['date'])
    df = df[(df['hour'] >= 0) & (df['hour'] <= 23)]
    if year is not None:
        df = df[df['date'].dt.year == year]
    df['season'] = df['date'].dt.month.map(_SEASON_MAP_OPT2)
    df['daytype'] = np.where(df['date'].dt.weekday < 5, 'wd', 'we')

    def _agg(sub):
        up_r = np.zeros(24); up_p = np.zeros(24)
        dn_r = np.zeros(24); dn_p = np.zeros(24)
        for h in range(24):
            s = sub[sub['hour'] == h]
            if len(s) == 0:
                continue
            up_r[h] = float((s['p_buy'] > 0).mean())
            au = s[s['p_buy'] > 0]
            up_p[h] = float(au['p_buy'].mean()) if len(au) else 0.0
            dn_r[h] = float((s['p_sell'] > 0).mean())
            ad = s[s['p_sell'] > 0]
            dn_p[h] = float(ad['p_sell'].mean()) if len(ad) else 0.0
        return up_r, up_p, dn_r, dn_p

    fb = _agg(df)
    out = {
        'fallback_upward_award_rate': fb[0],
        'fallback_upward_price_when_active': fb[1],
        'fallback_downward_award_rate': fb[2],
        'fallback_downward_price_when_active': fb[3],
        'buckets': {},
    }
    for season in ['W', 'Sp', 'Su', 'A']:
        for dtp in ['wd', 'we']:
            sub = df[(df['season'] == season) & (df['daytype'] == dtp)]
            up_r, up_p, dn_r, dn_p = _agg(sub)
            for h in range(24):
                if len(sub[sub['hour'] == h]) < min_cell_samples:
                    up_r[h], up_p[h] = fb[0][h], fb[1][h]
                    dn_r[h], dn_p[h] = fb[2][h], fb[3][h]
            out['buckets'][(season, dtp)] = {
                'upward_award_rate': up_r, 'upward_price_when_active': up_p,
                'downward_award_rate': dn_r, 'downward_price_when_active': dn_p,
            }
    return out


def _bucket_for_date(dt: datetime):
    season = _SEASON_MAP_OPT2[dt.month]
    dtp = 'wd' if dt.weekday() < 5 else 'we'
    return (season, dtp)


def build_overrides_from_conditioned_profiles(
    conditioned: Dict,
    start_date: datetime,
    end_date: datetime,
    mfrr_award_scale: float = 0.6,
    directional: bool = False,
) -> Dict[str, np.ndarray]:
    """Like build_overrides_from_msd_profiles, but selects the (season, daytype)
    bucket of the ACTUAL calendar day for every hour in the window, so award
    rates and active prices vary day-by-day instead of using one yearly mean.

    Drop-in: same return keys as build_overrides_from_msd_profiles for both
    legacy and directional modes; mutex normalisation identical.
    """
    n_days = (end_date - start_date).days
    n_hours = n_days * 24
    buckets = conditioned['buckets']

    def prof_for(dt):
        b = buckets.get(_bucket_for_date(dt))
        if b is None:  # safety fallback to hour-only marginal
            return (conditioned['fallback_upward_award_rate'],
                    conditioned['fallback_upward_price_when_active'],
                    conditioned['fallback_downward_award_rate'],
                    conditioned['fallback_downward_price_when_active'])
        return (b['upward_award_rate'], b['upward_price_when_active'],
                b['downward_award_rate'], b['downward_price_when_active'])

    if not directional:
        afrr_award = np.zeros(n_hours); afrr_energy = np.zeros(n_hours)
        mfrr_award = np.zeros(n_hours); mfrr_energy = np.zeros(n_hours)
        for h in range(n_hours):
            dt = start_date + timedelta(hours=h)
            up_r, up_p, dn_r, dn_p = prof_for(dt)
            hh = dt.hour
            afrr_award[h] = up_r[hh]; afrr_energy[h] = up_p[hh]
            mfrr_award[h] = dn_r[hh] * mfrr_award_scale; mfrr_energy[h] = dn_p[hh]
        return {
            "afrr_award_overrides": afrr_award, "afrr_energy_overrides": afrr_energy,
            "mfrr_award_overrides": mfrr_award, "mfrr_energy_overrides": mfrr_energy,
        }

    # Directional: emit both directions, with the SAME mutex normalisation
    # (P_up_excl + P_dn_excl <= 1) as the original directional builder.
    keys = ["afrr_up", "afrr_dn", "mfrr_up", "mfrr_dn"]
    arr = {f"{k}_award_overrides": np.zeros(n_hours) for k in keys}
    arr.update({f"{k}_energy_overrides": np.zeros(n_hours) for k in keys})
    for h in range(n_hours):
        dt = start_date + timedelta(hours=h)
        up_r, up_p, dn_r, dn_p = prof_for(dt)
        hh = dt.hour
        p_up_raw, p_dn_raw = up_r[hh], dn_r[hh]
        total = p_up_raw + p_dn_raw
        if total <= 1.0:
            p_up_excl, p_dn_excl = p_up_raw, p_dn_raw
        else:
            p_up_excl, p_dn_excl = p_up_raw / total, p_dn_raw / total
        arr["afrr_up_award_overrides"][h] = p_up_excl
        arr["afrr_up_energy_overrides"][h] = up_p[hh]
        arr["afrr_dn_award_overrides"][h] = p_dn_excl
        arr["afrr_dn_energy_overrides"][h] = dn_p[hh]
        arr["mfrr_up_award_overrides"][h] = p_up_excl * mfrr_award_scale
        arr["mfrr_up_energy_overrides"][h] = up_p[hh]
        arr["mfrr_dn_award_overrides"][h] = p_dn_excl * mfrr_award_scale
        arr["mfrr_dn_energy_overrides"][h] = dn_p[hh]
    return arr