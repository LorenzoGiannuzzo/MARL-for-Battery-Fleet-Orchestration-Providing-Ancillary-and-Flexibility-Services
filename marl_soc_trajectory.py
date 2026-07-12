# -*- coding: utf-8 -*-
"""
marl_soc_trajectory.py  --  FIX 2: traiettoria SOC media flotta per giorno.

La pipeline registra gia' `soc_mean` ORARIO per ogni policy in
`result.trajectories[policy]["soc_mean"]` (record_trajectory=True e' gia'
attivo). Questo modulo aggrega orario->giornaliero (media su blocchi di 24h),
salva `soc_daily.csv` e disegna `soc_daily.png` con una linea per policy, cosi'
si vede a colpo d'occhio la deriva/saturazione del SOC (la firma del capacity
farming: il warm-start parte a qualita' BC e sale saturando).

USO (una riga nel launcher, dopo il salvataggio di daily_profits.csv):

    from marl_soc_trajectory import save_soc_daily
    save_soc_daily(result, out_dir)

Non dipende da marl_charts / marl_analysis: usa solo numpy + matplotlib (Agg).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ordine e label dei modelli da plottare (chiavi di result.trajectories)
_POLICIES = [
    ("milp",        "MILP oracle"),
    ("bc",          "BC policy"),
    ("ppo_bc",      "PPO BC-warmstart"),
    ("ppo_vanilla", "PPO vanilla"),
    ("random",      "Random"),
]


def _hourly_to_daily(soc_hourly: np.ndarray) -> np.ndarray:
    """Media su blocchi di 24 ore. Tronca un'eventuale coda parziale."""
    x = np.asarray(soc_hourly, dtype=np.float64).ravel()
    n_days = x.size // 24
    if n_days == 0:
        return x  # meno di un giorno: restituisci cosi' com'e'
    return x[: n_days * 24].reshape(n_days, 24).mean(axis=1)


def collect_daily_soc(trajectories: Optional[Dict[str, Dict[str, Any]]]
                      ) -> Dict[str, np.ndarray]:
    """Estrae {policy: soc_giornaliero} da result.trajectories, saltando
    le policy assenti o senza soc_mean."""
    out: Dict[str, np.ndarray] = {}
    if not trajectories:
        return out
    for key, _label in _POLICIES:
        traj = trajectories.get(key)
        if not traj:
            continue
        soc = traj.get("soc_mean")
        if soc is None or len(soc) == 0:
            continue
        out[key] = _hourly_to_daily(soc)
    return out


def save_soc_daily(result: Any, out_dir) -> Optional[Path]:
    """Salva soc_daily.csv + soc_daily.png dai trajectory di `result`.
    Robusto: se non ci sono traiettorie o manca matplotlib, non solleva,
    stampa solo un avviso. Restituisce il path del PNG (o None)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectories = getattr(result, "trajectories", None)
    daily = collect_daily_soc(trajectories)
    if not daily:
        print("[soc_traj] nessuna traiettoria SOC trovata "
              "(record_trajectory attivo? result.trajectories popolato?)")
        return None

    n_days = max(len(v) for v in daily.values())

    # ---- CSV ----
    present = [(k, lbl) for k, lbl in _POLICIES if k in daily]
    csv_path = out_dir / "soc_daily.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day"] + [k for k, _ in present])
        for d in range(n_days):
            row = [d + 1]
            for k, _ in present:
                v = daily[k]
                row.append(f"{v[d]:.5f}" if d < len(v) else "")
            w.writerow(row)
    print(f"[soc_traj] salvato: {csv_path}")

    # ---- PNG ----
    png_path = out_dir / "soc_daily.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        for k, lbl in present:
            v = daily[k]
            ax.plot(np.arange(1, len(v) + 1), v, label=lbl, linewidth=1.4)
        ax.axhspan(0.10, 0.90, color="grey", alpha=0.06, zorder=0)
        ax.set_xlabel("Giorno di test")
        ax.set_ylabel("SOC medio flotta")
        ax.set_title("Traiettoria SOC media giornaliera per policy (finestra di test)")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        print(f"[soc_traj] salvato: {png_path}")
    except Exception as exc:  # matplotlib assente o backend problematico
        print(f"[soc_traj] PNG non generato ({exc}); il CSV c'e' comunque.")
        return None

    return png_path


if __name__ == "__main__":
    # smoke test con dati sintetici: warm che sale (satura), MILP/BC piatti bassi
    rng = np.random.default_rng(0)
    H = 366 * 24

    class _R:  # finto result
        trajectories = {
            "milp":        {"soc_mean": 0.145 + 0.01 * rng.standard_normal(H)},
            "bc":          {"soc_mean": 0.146 + 0.01 * rng.standard_normal(H)},
            "ppo_bc":      {"soc_mean": np.clip(0.30 + np.linspace(0, 0.5, H)
                                                + 0.02 * rng.standard_normal(H), 0.1, 0.9)},
            "ppo_vanilla": {"soc_mean": 0.40 + 0.02 * rng.standard_normal(H)},
            "random":      {"soc_mean": 0.476 + 0.02 * rng.standard_normal(H)},
        }

    save_soc_daily(_R(), "/tmp/soc_test")
