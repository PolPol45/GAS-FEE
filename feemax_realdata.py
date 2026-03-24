#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
FEE MAX (multi-fattore) - dati reali + live (ETH)
Titoli/label in italiano + conversione in Euro.

Fonti dati:
- Etherscan: serie storica gas price (CSV pubblico)
- Cloudflare Ethereum RPC: base fee live ultimo blocco
- CoinGecko: prezzo ETH/EUR live
"""

from __future__ import annotations

import argparse
import base64
import math
from io import StringIO
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# -------------------------
# PARAMETRI PRINCIPALI
# -------------------------
LOOKBACK_DAYS = 1200          # >= 750 per EMA lungo + finestre rolling
HORIZON_DAYS = 5              # orizzonte per "massimo reale nei prossimi 5 giorni"
GAS_LIMIT = 21000             # gas per una tx semplice (puoi cambiare)
TARGET_COVER = 0.99           # target copertura (99%) indicato nella logica del modello

# Coefficienti (come documento)
BETA_DEFAULT = 0.3
GAMMA_DEFAULT = 1.4
ALPHA_BASE_DEFAULT = 2.33     # ~99 percentile normale
ALPHA_SENS_DEFAULT = 0.0      # sensibilita' tanh (calibrato)

# Soglie e pesi regime
REGIME_HIGH_TH = 1.2
REGIME_LOW_TH = 0.8
REGIME_HIGH_K = 0.5
REGIME_LOW_K = 0.2

# Pesi mu_ln_LONG
W_EMA90 = 0.6
W_EMA750 = 0.4

# K di sicurezza rispetto alla media (cap automatico su FeeMax)
K_START = 2.0
K_MAX = 6.0
K_STEP = 0.25
ROLL_MEAN_DAYS = 30

# Plot: ultimi anni
PLOT_YEARS = 3

# Scenari multi-parametro (beta, gamma, alpha_sens)
SCENARIOS = [
    {"name": "base_calibrato", "beta": 0.3, "gamma": 1.4, "alpha_sens": 0.0},
    {"name": "b0.2_g1.0_a0.0", "beta": 0.2, "gamma": 1.0, "alpha_sens": 0.0},
    {"name": "b0.2_g1.2_a0.25", "beta": 0.2, "gamma": 1.2, "alpha_sens": 0.25},
    {"name": "b0.2_g1.6_a0.5", "beta": 0.2, "gamma": 1.6, "alpha_sens": 0.5},
    {"name": "b0.3_g1.0_a0.25", "beta": 0.3, "gamma": 1.0, "alpha_sens": 0.25},
    {"name": "b0.3_g1.2_a0.5", "beta": 0.3, "gamma": 1.2, "alpha_sens": 0.5},
    {"name": "b0.3_g1.6_a0.25", "beta": 0.3, "gamma": 1.6, "alpha_sens": 0.25},
    {"name": "b0.4_g1.0_a0.5", "beta": 0.4, "gamma": 1.0, "alpha_sens": 0.5},
    {"name": "b0.4_g1.2_a0.0", "beta": 0.4, "gamma": 1.2, "alpha_sens": 0.0},
    {"name": "b0.4_g1.4_a0.25", "beta": 0.4, "gamma": 1.4, "alpha_sens": 0.25},
    {"name": "b0.6_g1.2_a0.0", "beta": 0.6, "gamma": 1.2, "alpha_sens": 0.0},
    {"name": "b0.6_g1.6_a0.5", "beta": 0.6, "gamma": 1.6, "alpha_sens": 0.5},
]

# Dimensioni grafici (ridotte)
FIGSIZE_LONG = (10, 4)
FIGSIZE_SHORT = (10, 3.5)
FIGSIZE_SCATTER = (8, 4.5)

# -------------------------
# HELPER: formattazione "italiana"
# -------------------------
def fmt_num_it(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{x:,.{digits}f}"


def fmt_pct_it(x: float, digits: int = 2) -> str:
    return fmt_num_it(100 * x, digits) + "%"


# -------------------------
# 1) PREZZO ETH/EUR LIVE (CoinGecko)
# -------------------------
def fetch_eth_eur() -> Optional[float]:
    cg_url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=eur"
    try:
        r = requests.get(cg_url, timeout=20, headers={"User-Agent": "feemax-realdata/1.0"})
        r.raise_for_status()
        return float(r.json()["ethereum"]["eur"])
    except Exception as exc:
        print("Warning: unable to read ETH/EUR from CoinGecko. Error:", exc)
        return None


# -------------------------
# 2) SERIE STORICA GIORNALIERA (Etherscan CSV pubblico)
# -------------------------
def fetch_gas_series(lookback_days: int) -> pd.DataFrame:
    csv_url = "https://etherscan.io/chart/gasprice?output=csv"
    resp = requests.get(csv_url, timeout=30, headers={"User-Agent": "feemax-realdata/1.0"})
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    # colonne tipiche: Date(UTC), UnixTimeStamp, Value (Wei)
    df["Data"] = pd.to_datetime(df["Date(UTC)"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.rename(columns={"Value (Wei)": "FeeWei"})
    df["FeeWei"] = pd.to_numeric(df["FeeWei"], errors="coerce")
    df = df.dropna(subset=["Data", "FeeWei"]).sort_values("Data")

    # prendo solo lookback
    df = df[df["Data"] >= (df["Data"].max() - pd.Timedelta(days=lookback_days))].copy()

    # conversioni
    df["FeeGwei_reale"] = df["FeeWei"] / 1e9
    # evito zeri/negativi per il log
    df = df[df["FeeGwei_reale"] > 0].copy()
    return df


# -------------------------
# 3) DATO LIVE: BASE FEE ULTIMO BLOCCO (Cloudflare JSON-RPC)
# -------------------------
def _rpc_call(url: str, payload: dict) -> dict:
    r = requests.post(url, json=payload, timeout=20, headers={"User-Agent": "feemax-realdata/1.0"})
    r.raise_for_status()
    return r.json()


def fetch_live_basefee_gwei() -> Optional[float]:
    # RPC pubblici senza API key (fallback automatico)
    rpc_urls = [
        "https://cloudflare-eth.com",
        "https://ethereum.publicnode.com",
        "https://rpc.ankr.com/eth",
    ]

    for rpc_url in rpc_urls:
        try:
            # 1) Provo eth_feeHistory (piu' diretto per base fee)
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_feeHistory",
                "params": ["0x1", "latest", []],
            }
            j = _rpc_call(rpc_url, payload)
            if "result" in j and "baseFeePerGas" in j["result"]:
                base_fees = j["result"]["baseFeePerGas"]
                live_basefee_wei = int(base_fees[-1], 16)
                return live_basefee_wei / 1e9

            # 2) Fallback: eth_getBlockByNumber (legge baseFeePerGas dal blocco)
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getBlockByNumber",
                "params": ["latest", False],
            }
            j = _rpc_call(rpc_url, payload)
            if "result" in j and j["result"] and "baseFeePerGas" in j["result"]:
                live_basefee_wei = int(j["result"]["baseFeePerGas"], 16)
                return live_basefee_wei / 1e9
        except Exception:
            continue

    print("Warning: unable to read the live base fee from public RPC endpoints.")
    return None


def regime_adj(r: float) -> float:
    if pd.isna(r):
        return np.nan
    if r > REGIME_HIGH_TH:
        return REGIME_HIGH_K * math.log(r)
    if r < REGIME_LOW_TH:
        return REGIME_LOW_K * math.log(r)
    return 0.0


def alpha_t(
    sigma: float,
    sigma_med: float,
    alpha_base: float = ALPHA_BASE_DEFAULT,
    alpha_sens: float = ALPHA_SENS_DEFAULT,
) -> float:
    if pd.isna(sigma) or sigma_med is None or np.isnan(sigma_med) or sigma_med <= 0:
        return np.nan
    x = (sigma / sigma_med) - 1.0
    return alpha_base * (1.0 + alpha_sens * math.tanh(x))


# -------------------------
# 4) FEATURE DEL MODELLO (multi-fattore)
# -------------------------
def compute_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    out = df.copy()

    out["ln_fee"] = np.log(out["FeeGwei_reale"])

    # EMA su ln(fee): uso halflife come indicato (90 e 750 giorni)
    out["EMA90"] = out["ln_fee"].ewm(halflife=90, adjust=False).mean()
    out["EMA750"] = out["ln_fee"].ewm(halflife=750, adjust=False).mean()

    out["mu_ln_LONG"] = W_EMA90 * out["EMA90"] + W_EMA750 * out["EMA750"]

    # Volatilita' a 5 giorni: log_ret_5d e EWMA180 std
    out["log_ret_5d"] = out["ln_fee"] - out["ln_fee"].shift(5)
    out["sigma_d5"] = out["log_ret_5d"].ewm(halflife=180, adjust=False).std()

    # Momentum: ROC20, ROC40, accelerazione, z-score su 500gg
    out["ROC20"] = out["ln_fee"] - out["ln_fee"].shift(20)
    out["ROC40"] = out["ln_fee"] - out["ln_fee"].shift(40)
    out["Accel"] = out["ROC20"] - (out["ROC40"] - out["ROC20"])  # = 2*ROC20 - ROC40
    out["Momentum_raw"] = 0.6 * out["ROC20"] + 0.4 * out["Accel"]

    roll_win = 500
    out["Mom_mean500"] = out["Momentum_raw"].rolling(roll_win, min_periods=250).mean()
    out["Mom_std500"] = out["Momentum_raw"].rolling(roll_win, min_periods=250).std()
    out["MOMENTUM"] = (out["Momentum_raw"] - out["Mom_mean500"]) / out["Mom_std500"]

    # Regime adjustment: ratio = exp(EMA90-EMA750)
    out["ratio_regime"] = np.exp(out["EMA90"] - out["EMA750"])
    out["REGIME_ADJ"] = out["ratio_regime"].apply(regime_adj)

    # alpha_t dinamico (serve solo la mediana)
    sigma_med = out["sigma_d5"].median(skipna=True)
    return out, sigma_med


def compute_feemax(
    df: pd.DataFrame,
    sigma_med: float,
    alpha_base: float,
    alpha_sens: float,
    beta: float,
    gamma: float,
) -> pd.Series:
    sigma = df["sigma_d5"]
    x = (sigma / sigma_med) - 1.0
    alpha_dyn = alpha_base * (1.0 + alpha_sens * np.tanh(x))
    fee_ln = (
        df["mu_ln_LONG"]
        + alpha_dyn * sigma
        + beta * np.maximum(0, df["MOMENTUM"])
        + gamma * df["REGIME_ADJ"]
    )
    return np.exp(fee_ln)


def compute_fee_series(
    df: pd.DataFrame,
    sigma_med: float,
    alpha_base: float,
    alpha_sens: float,
    beta: float,
    gamma: float,
) -> pd.Series:
    return compute_feemax(df, sigma_med, alpha_base, alpha_sens, beta, gamma)


# -------------------------
# 5) TARGET DI TEST: massimo reale nei prossimi 5 giorni
# -------------------------
def compute_future_max(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MaxReale_prossimi5g_gwei"] = (
        out["FeeGwei_reale"][::-1]
        .rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS)
        .max()[::-1]
    )
    return out


# -------------------------
# 8) METRICHE (PScore, ecc.)
# -------------------------
def calc_metrics_from_series(df: pd.DataFrame, fee_series: pd.Series) -> Tuple[dict, pd.DataFrame]:
    tmp = df.copy()
    tmp["FeeMax_tmp"] = fee_series
    t = tmp.dropna(subset=["FeeMax_tmp", "MaxReale_prossimi5g_gwei", "FeeGwei_reale"]).copy()

    ratio = t["FeeMax_tmp"] / t["MaxReale_prossimi5g_gwei"]
    ln_ratio = np.log(ratio)

    success = t["FeeMax_tmp"] >= t["MaxReale_prossimi5g_gwei"]
    pscore = success.mean()
    overrun_rate = 1.0 - pscore

    overrun_ratio = t["MaxReale_prossimi5g_gwei"] / t["FeeMax_tmp"]
    sev = (overrun_ratio[~success] - 1.0).mean() if (~success).any() else 0.0
    worst = overrun_ratio.max()

    metrics = {
        "pscore": pscore,
        "overrun_rate": overrun_rate,
        "sev": sev,
        "worst": worst,
        "ratio": ratio,
        "ln_ratio": ln_ratio,
    }
    return metrics, t


def compute_metrics_summary(
    df: pd.DataFrame,
    fee_series: pd.Series,
    eth_eur: Optional[float],
) -> dict:
    metrics, t = calc_metrics_from_series(df, fee_series)

    diff = t["FeeMax_tmp"] - t["MaxReale_prossimi5g_gwei"]
    sd_diff_gwei = diff.std()

    avg_cost_eur = None
    if eth_eur:
        avg_cost_eur = (GAS_LIMIT * t["FeeMax_tmp"] * 1e-9 * eth_eur).mean()

    return {
        "PScore": metrics["pscore"],
        "SD_ln_ratio": metrics["ln_ratio"].std(),
        "SD_diff_gwei": sd_diff_gwei,
        "WorstOverrun": metrics["worst"],
        "CostoMedio_FeeMax_EUR": avg_cost_eur,
    }


def get_plotter(headless: bool):
    import matplotlib

    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_gwei(df_plot: pd.DataFrame, plt, title: str, lines: list, figsize=FIGSIZE_LONG):
    fig = plt.figure(figsize=figsize)
    plt.plot(
        df_plot["Data"],
        df_plot["FeeGwei_reale"],
        label="Observed fee (gwei)",
        linewidth=1.2,
    )
    for series, label, style in lines:
        plt.plot(df_plot["Data"], series, label=label, **style)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("gwei")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_eur(df_plot: pd.DataFrame, plt, title: str, figsize=FIGSIZE_LONG):
    fig = plt.figure(figsize=figsize)
    plt.plot(
        df_plot["Data"],
        df_plot["CostoTx_EUR_reale"],
        label=f"Observed cost (EUR) - gas={GAS_LIMIT}",
    )
    plt.plot(
        df_plot["Data"],
        df_plot["CostoTx_EUR_MaxReale5g"],
        label=f"Realized forward {HORIZON_DAYS}-day max (EUR)",
    )
    plt.plot(
        df_plot["Data"],
        df_plot["CostoTx_EUR_FeeMax"],
        label="Estimated FeeMax cost (EUR)",
    )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("EUR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


def print_ascii_table(title: str, headers: list, rows: list) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"

    print(f"\n{title}")
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"
        print(line)
    print(sep)


def build_results_rows(metrics: dict) -> list:
    rows = [
        ("PScore (copertura su max 5g)", metrics["pscore"]),
        ("OverrunRate (1 - PScore)", metrics["overrun_rate"]),
        ("Calibration (PScore - 99%)", metrics["pscore"] - TARGET_COVER),
        ("Severita'Overrun media (solo fallimenti)", metrics["sev"]),
        ("BiasScore (media ln(FeeMax/MaxReale))", metrics["ln_ratio"].mean()),
        ("WorstOverrun (max MaxReale/FeeMax)", metrics["worst"]),
        ("SD (dev.std ln(FeeMax/MaxReale))", metrics["ln_ratio"].std()),
        ("Headroom mediana (FeeMax/MaxReale)", metrics["ratio"].median()),
        ("Headroom media (FeeMax/MaxReale)", metrics["ratio"].mean()),
    ]

    pairs = []
    i = 0
    while i < len(rows):
        left = rows[i]
        right = rows[i + 1] if i + 1 < len(rows) else ("", None)
        pairs.append(
            {
                "Metrica": left[0],
                "Valore": fmt_num_it(left[1], 4),
                "Metrica ": right[0],
                "Valore ": fmt_num_it(right[1], 4) if right[1] is not None else "",
            }
        )
        i += 2

    headers = ["Metrica", "Valore", "Metrica", "Valore"]
    rows_out = []
    for p in pairs:
        rows_out.append([p["Metrica"], p["Valore"], p["Metrica "], p["Valore "]])
    return headers, rows_out


def print_results_table(metrics: dict) -> Tuple[list, list]:
    headers, rows_out = build_results_rows(metrics)
    print_ascii_table("=== RISULTATI TEST ===", headers, rows_out)
    return headers, rows_out

    print_ascii_table("=== RISULTATI TEST ===", headers, rows_out)


def print_brief_description(
    metrics: dict,
    eth_eur: Optional[float],
    df_test: pd.DataFrame,
    k_used: Optional[float],
) -> list:
    pscore = metrics["pscore"]
    overrun_rate = metrics["overrun_rate"]
    worst = metrics["worst"]
    headroom_mean = metrics["ratio"].mean()

    if pscore >= 0.99 and overrun_rate <= 0.01:
        rating = "Reliable"
    elif pscore >= 0.97:
        rating = "Moderate"
    else:
        rating = "Low"

    if headroom_mean >= 5:
        profile = "very conservative"
    elif headroom_mean >= 2:
        profile = "conservative"
    else:
        profile = "aggressive"

    line1 = (
        f"PScore coverage {fmt_pct_it(pscore, 2)} "
        f"(target {int(TARGET_COVER * 100)}%)."
    )
    line2 = f"OverrunRate {fmt_pct_it(overrun_rate, 2)}; worst overrun {fmt_num_it(worst, 4)}x."
    line3 = f"Average headroom {fmt_num_it(headroom_mean, 4)}x versus the realized 5-day maximum."

    lines = []
    lines.append(f"Assessment: {rating} ({profile}).")
    lines.append(line1)
    lines.append(line2)
    lines.append(line3)

    if eth_eur:
        avg_cost_eur = (GAS_LIMIT * df_test["FeeMax_tmp"] * 1e-9 * eth_eur).mean()
        lines.append(f"Average estimated FeeMax cost: {fmt_num_it(avg_cost_eur, 2)} EUR.")

    if k_used is not None:
        lines.append(
            f"Safety cap applied: K={fmt_num_it(k_used, 2)}x the {ROLL_MEAN_DAYS}-day rolling mean."
        )

    print("\n=== Summary ===")
    for ln in lines:
        print(ln)
    return lines


def print_data_checks(
    df: pd.DataFrame, eth_eur: Optional[float], live_basefee_gwei: Optional[float]
) -> list:
    last_date = df["Data"].max()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    age_days = (now - last_date).days if last_date is not None else None

    lines = []
    if last_date is not None:
        lines.append(
            f"Etherscan CSV: {len(df)} rows, latest date {last_date.date()}, age {age_days} days."
        )
        if age_days is not None and age_days > 2:
            lines.append("WARNING: Etherscan data has not updated in the last 2 days.")
    else:
        lines.append("WARNING: latest Etherscan date is unavailable.")

    if eth_eur is not None and eth_eur > 0:
        lines.append(f"CoinGecko ETH/EUR: {fmt_num_it(eth_eur, 2)} EUR.")
    else:
        lines.append("WARNING: ETH/EUR price is unavailable.")

    if live_basefee_gwei is not None:
        lines.append(f"Base fee RPC live: {fmt_num_it(live_basefee_gwei, 2)} gwei.")
    else:
        lines.append("WARNING: live base fee is unavailable.")

    print("\n=== Live data checks ===")
    for ln in lines:
        print(ln)
    return lines


def compute_mean_fee(df: pd.DataFrame, window_days: int) -> pd.Series:
    mean_roll = df["FeeGwei_reale"].rolling(window_days, min_periods=max(5, window_days // 3)).mean()
    return mean_roll.fillna(df["FeeGwei_reale"].expanding().mean())


def cap_with_k(df: pd.DataFrame, fee_series: pd.Series, k: float, mean_col: str) -> pd.Series:
    cap = k * df[mean_col]
    return fee_series.where(cap.isna(), np.minimum(fee_series, cap))


def apply_k_cap_search(
    df: pd.DataFrame,
    fee_series: pd.Series,
    mean_col: str,
    k_start: float = K_START,
    k_max: float = K_MAX,
    k_step: float = K_STEP,
    target: float = TARGET_COVER,
) -> Tuple[pd.Series, Optional[float], float]:
    best = None
    best_pscore = -1.0

    k = k_start
    while k <= k_max + 1e-9:
        capped = cap_with_k(df, fee_series, k, mean_col)
        metrics, _ = calc_metrics_from_series(df, capped)
        pscore = metrics["pscore"]
        if pscore > best_pscore:
            best = (capped, k, pscore)
            best_pscore = pscore
        if pscore >= target:
            return capped, k, pscore
        k += k_step

    if best is not None:
        return best[0], None, best[2]
    return fee_series, None, 0.0


def calc_leftover_eur(df_window: pd.DataFrame, fee_col: str, eth_eur: Optional[float]) -> Optional[float]:
    if eth_eur is None:
        return None
    diff_gwei = (df_window[fee_col] - df_window["FeeGwei_reale"]).clip(lower=0)
    diff_eur = GAS_LIMIT * diff_gwei * 1e-9 * eth_eur
    return diff_eur.mean()


def calc_failure_stats(
    df_window: pd.DataFrame, fee_col: str, eth_eur: Optional[float]
) -> Tuple[int, Optional[float], Optional[float], Optional[float]]:
    # fallimento: FeeMax < MaxReale_prossimi5g
    shortfall_gwei = (df_window["MaxReale_prossimi5g_gwei"] - df_window[fee_col]).clip(lower=0)
    fail_days = int((shortfall_gwei > 0).sum())
    if eth_eur is None:
        return fail_days, None, None, None
    shortfall_eur = GAS_LIMIT * shortfall_gwei * 1e-9 * eth_eur
    mean_eur = shortfall_eur[shortfall_eur > 0].mean() if fail_days > 0 else 0.0
    max_eur = shortfall_eur.max() if fail_days > 0 else 0.0
    total_eur_10tx = shortfall_eur.sum() * 10
    return fail_days, mean_eur, max_eur, total_eur_10tx


def calibrate_parameters(df: pd.DataFrame, sigma_med: float) -> Tuple[dict, dict]:
    alpha_base_vals = [2.0, 2.2, 2.33, 2.5, 2.7]
    alpha_sens_vals = [0.0, 0.15, 0.25, 0.35, 0.5]
    beta_vals = [0.2, 0.3, 0.4, 0.5, 0.6]
    gamma_vals = [0.6, 0.8, 1.0, 1.2, 1.4]

    best_params = None
    best_metrics = None
    best_score = None

    for a0 in alpha_base_vals:
        for asens in alpha_sens_vals:
            for b in beta_vals:
                for g in gamma_vals:
                    fee_series = compute_feemax(df, sigma_med, a0, asens, b, g)
                    metrics, _ = calc_metrics_from_series(df, fee_series)
                    pscore = metrics["pscore"]
                    headroom = metrics["ratio"].mean()
                    overrun_rate = metrics["overrun_rate"]

                    # Obiettivo: copertura vicina al target e headroom contenuto
                    penalty = abs(pscore - TARGET_COVER) * 500.0
                    if pscore < TARGET_COVER:
                        penalty += (TARGET_COVER - pscore) * 2000.0
                    penalty += headroom
                    penalty += overrun_rate * 10.0

                    if best_score is None or penalty < best_score:
                        best_score = penalty
                        best_params = {
                            "alpha_base": a0,
                            "alpha_sens": asens,
                            "beta": b,
                            "gamma": g,
                        }
                        best_metrics = metrics

    return best_params, best_metrics


def print_param_table(params: dict) -> Tuple[list, list]:
    headers = ["Parameter", "Calibrated value"]
    rows = [
        ["alpha_base", fmt_num_it(params["alpha_base"], 4)],
        ["alpha_sens", fmt_num_it(params["alpha_sens"], 4)],
        ["beta", fmt_num_it(params["beta"], 4)],
        ["gamma", fmt_num_it(params["gamma"], 4)],
    ]
    print_ascii_table("=== Calibrated parameters (public market data) ===", headers, rows)
    return headers, rows


def html_table(headers: list, rows: list) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for row in rows:
        tds = "".join(f"<td>{c}</td>" for c in row)
        body.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return data


def save_report(
    path: Path,
    data_checks: list,
    param_headers: list,
    param_rows: list,
    results_headers: list,
    results_rows: list,
    sens_headers: list,
    sens_rows: list,
    failure_headers: list,
    failure_rows: list,
    description_lines: list,
    fig_entries: list,
    settings: dict,
) -> None:
    css = """
    body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
    h1, h2 { margin: 8px 0 12px; }
    .meta { font-size: 12px; color: #444; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0 18px; }
    th, td { border: 1px solid #bbb; padding: 6px 8px; text-align: left; font-size: 13px; }
    th { background: #f0f0f0; }
    .section { margin: 18px 0; }
    .img { margin: 10px 0 22px; }
    .note { font-size: 12px; color: #555; }
    img { max-width: 900px; height: auto; }
    """

    imgs = []
    for title, fig in fig_entries:
        data = fig_to_base64(fig)
        imgs.append(
            f"<div class='img'><div class='note'>{title}</div>"
            f"<img src='data:image/png;base64,{data}' alt='{title}'/></div>"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>FeeMax Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>FeeMax Report</h1>
  <div class="meta">Generated: {settings.get('timestamp')}</div>
  <div class="meta">Chart window: last {settings.get('plot_years')} years; detail view {settings.get('last_days')} days</div>
  <div class="meta">Gas limit: {settings.get('gas_limit')} | K cap: {settings.get('k_used')}</div>

  <div class="section">
    <h2>Live data checks</h2>
    <ul>{"".join(f"<li>{ln}</li>" for ln in data_checks)}</ul>
  </div>

  <div class="section">
    <h2>Calibrated parameters</h2>
    {html_table(param_headers, param_rows)}
  </div>

  <div class="section">
    <h2>Backtest results</h2>
    {html_table(results_headers, results_rows)}
  </div>

  <div class="section">
    <h2>Sensitivity analysis</h2>
    {html_table(sens_headers, sens_rows)}
  </div>

  <div class="section">
    <h2>Failure-case diagnostics</h2>
    {html_table(failure_headers, failure_rows)}
  </div>

  <div class="section">
    <h2>Interpretation</h2>
    <ul>{"".join(f"<li>{ln}</li>" for ln in description_lines)}</ul>
  </div>

  <div class="section">
    <h2>Charts</h2>
    {''.join(imgs)}
  </div>

  <div class="section">
    <h2>Data sources</h2>
    <ul>
      <li>Etherscan CSV gas price (historical)</li>
      <li>CoinGecko ETH/EUR (live)</li>
      <li>Public Ethereum RPC (live base fee)</li>
    </ul>
  </div>
</body>
</html>
"""

    path.write_text(html, encoding="utf-8")
def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-factor FeeMax analysis on public Ethereum gas data.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening plot windows; still saves PNG charts.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save charts as PNG files even in GUI mode.",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Output directory for saved charts (default: plots).",
    )
    parser.add_argument(
        "--last-days",
        type=int,
        default=10,
        help="Add a detail chart for the last N days (0 disables it).",
    )
    parser.add_argument(
        "--plot-eur",
        action="store_true",
        help="Include EUR-denominated transaction cost charts.",
    )
    args = parser.parse_args()

    headless = args.headless
    save_plots = args.save_plots or headless
    plots_dir = Path(args.plots_dir)
    last_days = max(0, int(args.last_days))
    plot_eur_enabled = args.plot_eur

    plt = get_plotter(headless)

    eth_eur = fetch_eth_eur()
    if eth_eur is not None:
        print(f"Live ETH/EUR price: {fmt_num_it(eth_eur, 2)} EUR")
    else:
        print("Live ETH/EUR price unavailable.")

    df = fetch_gas_series(LOOKBACK_DAYS)
    if df.empty:
        raise SystemExit("No historical data available from Etherscan.")

    live_basefee_gwei = fetch_live_basefee_gwei()
    if live_basefee_gwei is not None:
        print(f"Live base fee (latest block): {fmt_num_it(live_basefee_gwei, 2)} gwei")
    else:
        print("Live base fee unavailable.")

    df_model, sigma_med = compute_features(df)
    df_model = compute_future_max(df_model)
    df_model["MeanFee30"] = compute_mean_fee(df_model, ROLL_MEAN_DAYS)

    data_checks_lines = print_data_checks(df, eth_eur, live_basefee_gwei)

    # -------------------------
    # 5b) CALIBRAZIONE PARAMETRI (da dati reali)
    # -------------------------
    params, _ = calibrate_parameters(df_model, sigma_med)
    if params is None:
        params = {
            "alpha_base": ALPHA_BASE_DEFAULT,
            "alpha_sens": ALPHA_SENS_DEFAULT,
            "beta": BETA_DEFAULT,
            "gamma": GAMMA_DEFAULT,
        }
    param_headers, param_rows = print_param_table(params)

    # FeeMax base + cap K
    fee_base = compute_feemax(
        df_model,
        sigma_med,
        params["alpha_base"],
        params["alpha_sens"],
        params["beta"],
        params["gamma"],
    )
    df_model["FeeMax_base"] = fee_base
    fee_final, k_used, _ = apply_k_cap_search(df_model, fee_base, "MeanFee30")
    df_model["FeeMax_final"] = fee_final

    # Metriche su FeeMax finale (cappata)
    metrics, df_test = calc_metrics_from_series(df_model, df_model["FeeMax_final"])

    # -------------------------
    # 6) CONVERSIONE IN EURO (costo tx con GAS_LIMIT)
    # -------------------------
    if eth_eur:
        df_test["CostoTx_EUR_reale"] = GAS_LIMIT * df_test["FeeGwei_reale"] * 1e-9 * eth_eur
        df_test["CostoTx_EUR_FeeMax"] = GAS_LIMIT * df_test["FeeMax_tmp"] * 1e-9 * eth_eur
        df_test["CostoTx_EUR_MaxReale5g"] = (
            GAS_LIMIT * df_test["MaxReale_prossimi5g_gwei"] * 1e-9 * eth_eur
        )

    # -------------------------
    # 7) TABELLE RISULTATI
    # -------------------------
    results_headers, results_rows = print_results_table(metrics)

    # -------------------------
    # 8) MULTI-SCENARIO (beta, gamma, alpha_sens) + CSV
    # -------------------------
    scenario_rows = []
    scenario_cols = {}
    alpha_base = params["alpha_base"]

    for sc in SCENARIOS:
        fee_series = compute_feemax(
            df_model,
            sigma_med,
            alpha_base,
            sc["alpha_sens"],
            sc["beta"],
            sc["gamma"],
        )
        if k_used is not None:
            fee_series = cap_with_k(df_model, fee_series, k_used, "MeanFee30")

        metrics_sc, t_sc = calc_metrics_from_series(df_model, fee_series)
        headroom_mean = metrics_sc["ratio"].mean()
        headroom_median = metrics_sc["ratio"].median()
        cost_mean_eur = None
        if eth_eur:
            cost_mean_eur = (GAS_LIMIT * t_sc["FeeMax_tmp"] * 1e-9 * eth_eur).mean()

        scenario_rows.append(
            {
                "scenario": sc["name"],
                "alpha_base": alpha_base,
                "alpha_sens": sc["alpha_sens"],
                "beta": sc["beta"],
                "gamma": sc["gamma"],
                "PScore": metrics_sc["pscore"],
                "WorstOverrun": metrics_sc["worst"],
                "Headroom_mean": headroom_mean,
                "Headroom_median": headroom_median,
                "Costo_medio_EUR": cost_mean_eur,
            }
        )

        col = f"FeeMax_{sc['name']}"
        df_model[col] = fee_series
        scenario_cols[sc["name"]] = col

    results = pd.DataFrame(scenario_rows)
    # calcolo "leftover" medio in EUR (FeeMax - FeeReale, solo positivo)
    df_plot_all = df_model.dropna(subset=["FeeGwei_reale"]).copy()
    cutoff_3y_all = df_plot_all["Data"].max() - pd.Timedelta(days=365 * PLOT_YEARS)
    df_3y = df_plot_all[df_plot_all["Data"] >= cutoff_3y_all].copy()
    df_last5 = df_plot_all[df_plot_all["Data"] >= (df_plot_all["Data"].max() - pd.Timedelta(days=5))].copy()

    leftovers_3y = []
    leftovers_5d = []
    for _, r in results.iterrows():
        col = f"FeeMax_{r['scenario']}"
        if col in df_plot_all.columns:
            avg_left_3y = calc_leftover_eur(df_3y, col, eth_eur)
            avg_left_5d = calc_leftover_eur(df_last5, col, eth_eur)
        else:
            avg_left_3y = None
            avg_left_5d = None
        leftovers_3y.append(avg_left_3y)
        leftovers_5d.append(avg_left_5d)

    results["Avanzo_medio_EUR_3y"] = leftovers_3y
    results["Avanzo_medio_EUR_5d"] = leftovers_5d
    results["Avanzo_medio_EUR_3y_10tx"] = results["Avanzo_medio_EUR_3y"] * 10
    results["Avanzo_medio_EUR_5d_10tx"] = results["Avanzo_medio_EUR_5d"] * 10

    results.to_csv("feemax_scenarios_results.csv", index=False)
    print("Scenario metrics saved: feemax_scenarios_results.csv")

    # selezione 3 scenari: migliore (efficiente), piu conservativo, meno conservativo
    results_sorted = results.sort_values(["PScore", "Avanzo_medio_EUR_3y"], ascending=[False, True])
    best = results_sorted.iloc[0]["scenario"]
    conservative = results.sort_values(["PScore", "Headroom_mean"], ascending=[False, False]).iloc[0]["scenario"]
    aggressive = results.sort_values("Avanzo_medio_EUR_3y", ascending=True).iloc[0]["scenario"]

    # Forza la presenza dello scenario calibrato se esiste
    base_name = "base_calibrato"
    selected = []
    for name in [conservative, base_name, aggressive]:
        if name in results["scenario"].tolist() and name not in selected:
            selected.append(name)
    if len(selected) < 3:
        for name in results["scenario"].tolist():
            if name not in selected:
                selected.append(name)
            if len(selected) == 3:
                break

    selected_results = results[results["scenario"].isin(selected)].copy()
    role_map = {
        conservative: "most conservative",
        base_name: "calibrated",
        aggressive: "least conservative",
    }
    selected_results["Role"] = selected_results["scenario"].map(role_map).fillna("")
    role_order = {"most conservative": 0, "calibrated": 1, "least conservative": 2}
    selected_results["__role_ord"] = selected_results["Role"].map(role_order).fillna(99)
    selected_results = selected_results.sort_values("__role_ord").drop(columns=["__role_ord"])

    print(
        f"\nSelected scenarios -> most conservative: {conservative} | calibrated: {base_name} | least conservative: {aggressive}"
    )

    # tabella scenari (solo 3)
    scen_headers = [
        "Role",
        "Scenario",
        "beta",
        "gamma",
        "alpha_sens",
        "PScore",
        "WorstOverrun",
        "Headroom_mean",
        "Headroom_median",
        "Average EUR excess (10 tx, 3y)",
        "Average EUR excess (10 tx, 5d)",
    ]
    scen_rows = []
    for _, r in selected_results.iterrows():
        scen_rows.append(
            [
                r["Role"],
                r["scenario"],
                fmt_num_it(r["beta"], 2),
                fmt_num_it(r["gamma"], 2),
                fmt_num_it(r["alpha_sens"], 2),
                fmt_pct_it(r["PScore"], 2),
                fmt_num_it(r["WorstOverrun"], 4),
                fmt_num_it(r["Headroom_mean"], 4),
                fmt_num_it(r["Headroom_median"], 4),
                (fmt_num_it(r["Avanzo_medio_EUR_3y_10tx"], 2) + " EUR") if eth_eur else "",
                (fmt_num_it(r["Avanzo_medio_EUR_5d_10tx"], 2) + " EUR") if eth_eur else "",
            ]
        )
    print_ascii_table("=== Multi-scenario analysis (top 3) ===", scen_headers, scen_rows)
    sens_headers = scen_headers
    sens_rows = scen_rows

    # scostamenti nei casi di fallimento (solo 3 scenari)
    failure_headers = [
        "Role",
        "Scenario",
        "Window",
        "Fail tx (10x)",
        "Average EUR shortfall",
        "Max EUR shortfall",
        "Total EUR shortfall (10x)",
    ]
    failure_rows = []
    for _, r in selected_results.iterrows():
        col = f"FeeMax_{r['scenario']}"
        for label, win_df in [("3y", df_3y), ("5d", df_last5)]:
            if col in win_df.columns:
                fail_days, mean_eur, max_eur, total_eur_10 = calc_failure_stats(
                    win_df, col, eth_eur
                )
                failure_rows.append(
                    [
                        r["Role"],
                        r["scenario"],
                        label,
                        str(fail_days * 10),
                        (fmt_num_it(mean_eur, 2) + " EUR") if eth_eur else "",
                        (fmt_num_it(max_eur, 2) + " EUR") if eth_eur else "",
                        (fmt_num_it(total_eur_10, 2) + " EUR") if eth_eur else "",
                    ]
                )
    print_ascii_table("=== Failure-case diagnostics (10 tx) ===", failure_headers, failure_rows)

    # Nota: per richiesta attuale, mostriamo solo i 3 scenari selezionati nelle tabelle e nei grafici.

    # -------------------------
    # 9) DESCRIZIONE RISULTATI
    # -------------------------
    description_lines = print_brief_description(metrics, eth_eur, df_test, k_used)
    if live_basefee_gwei is not None and eth_eur:
        live_cost_eur = GAS_LIMIT * live_basefee_gwei * 1e-9 * eth_eur
        live_line = (
            f"Estimated LIVE cost from the latest block base fee for gas={GAS_LIMIT}: "
            f"{fmt_num_it(live_cost_eur, 2)} EUR"
        )
        print(live_line)
        description_lines.append(live_line)

    # -------------------------
    # 10) GRAFICI (3 anni + dettaglio ultimi giorni)
    # -------------------------
    df_plot = df_model.dropna(subset=["FeeMax_final", "FeeGwei_reale", "MaxReale_prossimi5g_gwei"]).copy()
    cutoff_3y = df_plot["Data"].max() - pd.Timedelta(days=365 * PLOT_YEARS)
    df_plot_3y = df_plot[df_plot["Data"] >= cutoff_3y].copy()

    report_figs = []

    # Unico grafico 3 anni nel report: fee_max_vs_reali_gwei.png (3 scenari + fee reale)

    # Grafico richiesto: fee_max_vs_reali_gwei.png (solo 3 scenari + fee reale)
    fig_fee = plt.figure(figsize=FIGSIZE_LONG)
    plt.plot(df_plot_3y["Data"], df_plot_3y["FeeGwei_reale"], label="Observed fee (gwei)", linewidth=1.4)
    for name in selected:
        col = scenario_cols.get(name)
        if col and col in df_plot_3y.columns:
            plt.plot(
                df_plot_3y["Data"],
                df_plot_3y[col],
                label=f"FeeMax {name}",
                linewidth=0.9,
                alpha=0.75,
            )
    plt.title("fee_max_vs_realized_gwei (3 scenarios)")
    plt.xlabel("Date")
    plt.ylabel("gwei")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_fee.savefig("fee_max_vs_reali_gwei.png", dpi=150)
    print("Chart saved: fee_max_vs_reali_gwei.png")
    report_figs.append(("fee_max_vs_realized_gwei (3 scenarios)", fig_fee))

    if last_days > 0:
        cutoff = df_plot["Data"].max() - pd.Timedelta(days=last_days)
        df_last = df_plot[df_plot["Data"] >= cutoff].copy()
        if not df_last.empty:
            lines_last = []
            for name in selected:
                col = scenario_cols.get(name)
                if col and col in df_last.columns:
                    lines_last.append((df_last[col], f"FeeMax ({name})", {"linewidth": 1.2, "alpha": 0.9}))
            title_last = f"Ethereum: last {last_days} days detail (gwei)"
            fig = plot_gwei(df_last, plt, title_last, lines_last, figsize=FIGSIZE_SHORT)
            report_figs.append((title_last, fig))

    # (Opzionale) Grafici in Euro
    if plot_eur_enabled and eth_eur:
        df_test_plot = df_test.copy()
        cutoff_3y_eur = df_test_plot["Data"].max() - pd.Timedelta(days=365 * PLOT_YEARS)
        df_test_3y = df_test_plot[df_test_plot["Data"] >= cutoff_3y_eur].copy()
        title_eur_3y = f"Ethereum: last {PLOT_YEARS} years (EUR, gas={GAS_LIMIT})"
        fig = plot_eur(df_test_3y, plt, title_eur_3y)
        report_figs.append((title_eur_3y, fig))
        if last_days > 0:
            cutoff = df_test_plot["Data"].max() - pd.Timedelta(days=last_days)
            df_last_eur = df_test_plot[df_test_plot["Data"] >= cutoff].copy()
            if not df_last_eur.empty:
                title_eur_last = f"Ethereum: last {last_days} days detail (EUR)"
                fig = plot_eur(df_last_eur, plt, title_eur_last, figsize=FIGSIZE_SHORT)
                report_figs.append((title_eur_last, fig))

    # Scatter PScore vs costo medio (EUR)
    if eth_eur:
        fig_scatter = plt.figure(figsize=FIGSIZE_SCATTER)
        sel_df = selected_results.copy()
        x = sel_df["Costo_medio_EUR"]
        y = sel_df["PScore"]
        plt.scatter(x, y, s=50, alpha=0.85)
        for _, r in sel_df.iterrows():
            label = f"b{r['beta']}_g{r['gamma']}_a{r['alpha_sens']}"
            plt.text(r["Costo_medio_EUR"], r["PScore"], label, fontsize=8)
        plt.title("pscore_vs_cost_scatter (3 scenarios)")
        plt.xlabel("Average FeeMax cost (EUR)")
        plt.ylabel("PScore")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_scatter.savefig("pscore_vs_cost_scatter.png", dpi=150)
        print("Chart saved: pscore_vs_cost_scatter.png")
        report_figs.append(("pscore_vs_cost_scatter (3 scenarios)", fig_scatter))
    else:
        print("Unable to create pscore_vs_cost_scatter.png: ETH/EUR unavailable.")

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"report_{timestamp}.html"
    settings = {
        "timestamp": timestamp,
        "plot_years": PLOT_YEARS,
        "last_days": last_days,
        "gas_limit": GAS_LIMIT,
        "k_used": fmt_num_it(k_used, 2) if k_used is not None else "n/a",
    }
    save_report(
        report_path,
        data_checks_lines,
        param_headers,
        param_rows,
        results_headers,
        results_rows,
        sens_headers,
        sens_rows,
        failure_headers,
        failure_rows,
        description_lines,
        report_figs,
        settings,
    )
    print(f"Report saved: {report_path}")

    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(plt.get_fignums(), start=1):
            out = plots_dir / f"feemax_plot_{i}.png"
            plt.figure(fig).savefig(out, dpi=150)
            print(f"Chart saved: {out}")

    if headless:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
