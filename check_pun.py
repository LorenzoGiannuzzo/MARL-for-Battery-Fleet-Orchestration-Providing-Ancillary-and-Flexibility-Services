# Mean day-ahead price per year, for the figures quoted in Section 2.
#
# The loader returns a TUPLE, not a bare array: the first element carries the
# timestamps and the second the prices, which is why treating the result as a
# single sequence raised a TypeError on datetime.
from italian_market_data import load_pun_from_gme_xlsx_multi
import numpy as np

for year, path in ((2023, r"data\20230101_20231231_PUN.xlsx"),
                   (2024, r"data\20240101_20241231_PUN.xlsx")):
    out = load_pun_from_gme_xlsx_multi([path], price_column="PUN")

    # Locate the numeric member of the tuple rather than assuming its position.
    prices = None
    for part in (out if isinstance(out, tuple) else (out,)):
        arr = np.asarray(list(part.values()) if isinstance(part, dict) else part)
        if arr.dtype.kind in "fiu":
            prices = arr.astype(float)
            break
    if prices is None:
        print(f"{year}: no numeric series found in {type(out)}")
        continue

    prices = prices[np.isfinite(prices)]
    print(f"{year}: mean {prices.mean():.2f} EUR/MWh, "
          f"std {prices.std():.2f}, n = {len(prices)}")
