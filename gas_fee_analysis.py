#!/usr/bin/env python3
"""Gas fee analysis (FeeMax) - script runnable outside Jupyter.

Keeps the same logic as the notebook, but removes notebook-only bits
(!pip, display) and saves plots to files instead of showing them.
"""

import sys
import math
from io import StringIO
from datetime import datetime

# ---- dependency checks ----
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import requests
except Exception as exc:
    print("Errore: dipendenze mancanti. Installa con:")
    print("  pip install pandas numpy matplotlib requests")
    print(f"Dettaglio errore: {exc}")
    sys.exit(0)

# -------------------------
# PARAMETRI PRINCIPALI
# -------------------------
LOOKBACK_DAYS = 1200          # >= 750 per EMA lungo + finestre rolling
HORIZON_DAYS  = 5             # orizzonte per "massimo reale nei prossimi 5 giorni"
GAS_LIMIT     = 21000         # gas per una tx semplice (puoi cambiare)
TARGET_COVER  = 0.99          # target copertura (99%) indicato nella logica del modello

# Coefficienti (come documento)
BETA_DEFAULT  = 0.4
GAMMA_DEFAULT = 1.0
ALPHA_BASE_DEFAULT = 2.33     # ~99° percentile normale
ALPHA_SENS_DEFAULT = 0.25     # sensibilità tanh

# Soglie e pesi regime
REGIME_HIGH_TH = 1.2
REGIME_LOW_TH  = 0.8
REGIME_HIGH_K  = 0.5
REGIME_LOW_K   = 0.2

# Pesi µ_ln_LONG
W_EMA90  = 0.6
W_EMA750 = 0.4

# -------------------------
# HELPER: formattazione "italiana"
# -------------------------
def fmt_num_it(x, digits=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    s = f"{x:,.{digits}f}"
    # da 1,234.56 -> 1.234,56
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def fmt_pct_it(x, digits=2):
    return fmt_num_it(100 * x, digits) + "%"


# -------------------------
# 1) PREZZO ETH/EUR LIVE (CoinGecko)
# -------------------------
eth_eur = None
cg_url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=eur"
try:
    r = requests.get(cg_url, timeout=20, headers={"User-Agent": "gas-fee-analysis/1.0"})
    r.raise_for_status()
    eth_eur = float(r.json()["ethereum"]["eur"])
except Exception as e:
    print("Attenzione: non riesco a leggere ETH/EUR da CoinGecko. Errore:", e)
    eth_eur = None

print(f"Prezzo ETH/EUR (live): {fmt_num_it(eth_eur,2)} €" if eth_eur else "Prezzo ETH/EUR non disponibile.")

# -------------------------
# 2) SERIE STORICA GIORNALIERA (Etherscan CSV pubblico)
# -------------------------
csv_url = "https://etherscan.io/chart/gasprice?output=csv"

try:
    resp = requests.get(csv_url, timeout=30, headers={"User-Agent": "gas-fee-analysis/1.0"})
    resp.raise_for_status()
except Exception as e:
    print("Errore: non riesco a scaricare la serie storica da Etherscan.")
    print("Dettaglio errore:", e)
    sys.exit(0)

df = pd.read_csv(StringIO(resp.text))
# colonne tipiche: Date(UTC), UnixTimeStamp, Value (Wei)
if "Date(UTC)" not in df.columns or "Value (Wei)" not in df.columns:
    print("Errore: il CSV non ha le colonne attese (Date(UTC), Value (Wei)).")
    print("Colonne trovate:", list(df.columns))
    sys.exit(0)

df["Data"] = pd.to_datetime(df["Date(UTC)"], errors="coerce", utc=True).dt.tz_convert(None)
df = df.rename(columns={"Value (Wei)": "FeeWei"})
df["FeeWei"] = pd.to_numeric(df["FeeWei"], errors="coerce")
df = df.dropna(subset=["Data", "FeeWei"]).sort_values("Data")

# prendo solo lookback
df = df[df["Data"] >= (df["Data"].max() - pd.Timedelta(days=LOOKBACK_DAYS))].copy()

# conversioni
df["FeeGwei_reale"] = df["FeeWei"] / 1e9
# evito zeri/negativi per il log
df = df[df["FeeGwei_reale"] > 0].copy()

# -------------------------
# 3) DATO LIVE: BASE FEE ULTIMO BLOCCO (Cloudflare JSON-RPC)
# -------------------------
rpc_url = "https://cloudflare-eth.com"
live_basefee_gwei = None

try:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_feeHistory",
        "params": ["0x1", "latest", []],
    }
    r = requests.post(rpc_url, json=payload, timeout=20, headers={"User-Agent": "gas-fee-analysis/1.0"})
    r.raise_for_status()
    j = r.json()
    base_fees = j["result"]["baseFeePerGas"]
    live_basefee_wei = int(base_fees[-1], 16)
    live_basefee_gwei = live_basefee_wei / 1e9
except Exception as e:
    print("Attenzione: non riesco a leggere la base fee live via RPC. Errore:", e)

if live_basefee_gwei is not None:
    print(f"Base fee live (ultimo blocco): {fmt_num_it(live_basefee_gwei,2)} gwei")
else:
    print("Base fee live non disponibile.")

# -------------------------
# 4) IMPLEMENTAZIONE FORMULA (multi-fattore)
#    ln(FeeMax) = µ_ln_LONG + α_t * σ_Δ5 + β*max(0,MOMENTUM) + γ*REGIME_ADJ
# -------------------------

df["ln_fee"] = np.log(df["FeeGwei_reale"])

# EMA su ln(fee): uso halflife come indicato (90 e 750 giorni)
df["EMA90"] = df["ln_fee"].ewm(halflife=90, adjust=False).mean()
df["EMA750"] = df["ln_fee"].ewm(halflife=750, adjust=False).mean()

df["mu_ln_LONG"] = W_EMA90 * df["EMA90"] + W_EMA750 * df["EMA750"]

# Volatilità a 5 giorni: log_ret_5d e EWMA180 std
df["log_ret_5d"] = df["ln_fee"] - df["ln_fee"].shift(5)
df["sigma_d5"] = df["log_ret_5d"].ewm(halflife=180, adjust=False).std()

# Momentum: ROC20, ROC40, accelerazione, z-score su 500gg
df["ROC20"] = df["ln_fee"] - df["ln_fee"].shift(20)
df["ROC40"] = df["ln_fee"] - df["ln_fee"].shift(40)
df["Accel"] = df["ROC20"] - (df["ROC40"] - df["ROC20"])  # = 2*ROC20 - ROC40
df["Momentum_raw"] = 0.6 * df["ROC20"] + 0.4 * df["Accel"]

roll_win = 500
df["Mom_mean500"] = df["Momentum_raw"].rolling(roll_win, min_periods=250).mean()
df["Mom_std500"] = df["Momentum_raw"].rolling(roll_win, min_periods=250).std()
df["MOMENTUM"] = (df["Momentum_raw"] - df["Mom_mean500"]) / df["Mom_std500"]

# Regime adjustment: ratio = exp(EMA90-EMA750)
df["ratio_regime"] = np.exp(df["EMA90"] - df["EMA750"])


def regime_adj(r):
    if pd.isna(r):
        return np.nan
    if r > REGIME_HIGH_TH:
        return REGIME_HIGH_K * math.log(r)
    if r < REGIME_LOW_TH:
        return REGIME_LOW_K * math.log(r)
    return 0.0


df["REGIME_ADJ"] = df["ratio_regime"].apply(regime_adj)

# α_t dinamico
sigma_med = df["sigma_d5"].median(skipna=True)


def alpha_t(sigma, alpha_base=ALPHA_BASE_DEFAULT, alpha_sens=ALPHA_SENS_DEFAULT):
    if pd.isna(sigma) or sigma_med is None or np.isnan(sigma_med) or sigma_med <= 0:
        return np.nan
    x = (sigma / sigma_med) - 1.0
    return alpha_base * (1.0 + alpha_sens * math.tanh(x))


df["alpha_t"] = df["sigma_d5"].apply(lambda s: alpha_t(s))

# FeeMax
df["FeeMax_ln"] = (
    df["mu_ln_LONG"]
    + df["alpha_t"] * df["sigma_d5"]
    + BETA_DEFAULT * np.maximum(0, df["MOMENTUM"])
    + GAMMA_DEFAULT * df["REGIME_ADJ"]
)

df["FeeMax_gwei"] = np.exp(df["FeeMax_ln"])

# -------------------------
# 5) TARGET DI TEST: massimo reale nei prossimi 5 giorni
# -------------------------
# per ogni t: max(Fee reale da t a t+4)
df["MaxReale_prossimi5g_gwei"] = (
    df["FeeGwei_reale"][::-1]
    .rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS)
    .max()
    [::-1]
)

# pulizia righe con NaN (inizio serie e ultimi giorni)
df_test = df.dropna(subset=["FeeMax_gwei", "MaxReale_prossimi5g_gwei", "FeeGwei_reale"]).copy()

# -------------------------
# 6) CONVERSIONE IN EURO (costo tx con GAS_LIMIT)
# -------------------------
if eth_eur:
    df_test["CostoTx_EUR_reale"] = GAS_LIMIT * df_test["FeeGwei_reale"] * 1e-9 * eth_eur
    df_test["CostoTx_EUR_FeeMax"] = GAS_LIMIT * df_test["FeeMax_gwei"] * 1e-9 * eth_eur
    df_test["CostoTx_EUR_MaxReale5g"] = GAS_LIMIT * df_test["MaxReale_prossimi5g_gwei"] * 1e-9 * eth_eur

# -------------------------
# 7) GRAFICO: Fee Max vs fee reali (gwei)
# -------------------------
plt.figure(figsize=(14, 6))
plt.plot(df_test["Data"], df_test["FeeGwei_reale"], label="Fee reale (gwei)")
plt.plot(df_test["Data"], df_test["MaxReale_prossimi5g_gwei"], label=f"Massimo reale nei prossimi {HORIZON_DAYS} giorni (gwei)")
plt.plot(df_test["Data"], df_test["FeeMax_gwei"], label="Fee Max stimata (gwei)")
plt.title("Ethereum: andamento Fee Max stimata vs fee reali")
plt.xlabel("Data")
plt.ylabel("gwei")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fee_max_vs_reali_gwei.png", dpi=150)

# (Opzionale) Grafico in Euro per una tx standard
if eth_eur:
    plt.figure(figsize=(14, 6))
    plt.plot(df_test["Data"], df_test["CostoTx_EUR_reale"], label=f"Costo reale (EUR) - gas={GAS_LIMIT}")
    plt.plot(df_test["Data"], df_test["CostoTx_EUR_MaxReale5g"], label=f"Massimo reale prossimi {HORIZON_DAYS}g (EUR)")
    plt.plot(df_test["Data"], df_test["CostoTx_EUR_FeeMax"], label=f"Costo stimato Fee Max (EUR)")
    plt.title(f"Ethereum: conversione in Euro (costo per transazione, gas={GAS_LIMIT})")
    plt.xlabel("Data")
    plt.ylabel("Euro (€)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fee_max_vs_reali_eur.png", dpi=150)

# -------------------------
# 8) SCORE MATRIX (PScore, ecc.)
# -------------------------
ratio = df_test["FeeMax_gwei"] / df_test["MaxReale_prossimi5g_gwei"]
ln_ratio = np.log(ratio)

success = df_test["FeeMax_gwei"] >= df_test["MaxReale_prossimi5g_gwei"]
pscore = success.mean()
overrun_rate = 1.0 - pscore

# severità
overrun_ratio = df_test["MaxReale_prossimi5g_gwei"] / df_test["FeeMax_gwei"]
sev = (overrun_ratio[~success] - 1.0).mean() if (~success).any() else 0.0
worst = overrun_ratio.max()

scores = pd.DataFrame(
    {
        "Valore": [
            pscore,
            pscore - TARGET_COVER,
            ln_ratio.mean(),
            ln_ratio.std(),
            overrun_rate,
            sev,
            worst,
            ratio.median(),
            ratio.mean(),
        ]
    },
    index=[
        f"PScore (copertura su max {HORIZON_DAYS}g)",
        f"Calibration (PScore - {int(TARGET_COVER * 100)}%)",
        "BiasScore (media ln(FeeMax/MaxReale))",
        "SD (dev.std ln(FeeMax/MaxReale))",
        "OverrunRate (1 - PScore)",
        "SeveritàOverrun media (solo fallimenti)",
        "WorstOverrun (max MaxReale/FeeMax)",
        "Headroom mediana (FeeMax/MaxReale)",
        "Headroom media (FeeMax/MaxReale)",
    ],
)

# format italiano
scores_fmt = scores.copy()
scores_fmt["Valore"] = scores_fmt["Valore"].apply(lambda x: fmt_num_it(x, 4))
print("\n=== Score Matrix ===")
print(scores_fmt.to_string())

# -------------------------
# 9) RATE DI SUCCESSO + SENSITIVITY ANALYSIS + SD (tabella)
# -------------------------

def compute_metrics(alpha_base, alpha_sens, beta, gamma):
    tmp = df.copy()

    # ricalcolo alpha e feemax
    tmp["alpha_t2"] = tmp["sigma_d5"].apply(
        lambda s: alpha_t(s, alpha_base=alpha_base, alpha_sens=alpha_sens)
    )
    tmp["FeeMax_ln2"] = (
        tmp["mu_ln_LONG"]
        + tmp["alpha_t2"] * tmp["sigma_d5"]
        + beta * np.maximum(0, tmp["MOMENTUM"])
        + gamma * tmp["REGIME_ADJ"]
    )
    tmp["FeeMax_gwei2"] = np.exp(tmp["FeeMax_ln2"])
    tmp["MaxReale_prossimi5g_gwei2"] = (
        tmp["FeeGwei_reale"][::-1]
        .rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS)
        .max()
        [::-1]
    )
    t = tmp.dropna(subset=["FeeMax_gwei2", "MaxReale_prossimi5g_gwei2"]).copy()
    suc = t["FeeMax_gwei2"] >= t["MaxReale_prossimi5g_gwei2"]
    p = suc.mean()

    r = t["FeeMax_gwei2"] / t["MaxReale_prossimi5g_gwei2"]
    lnr = np.log(r)
    sd_ln = lnr.std()

    diff = t["FeeMax_gwei2"] - t["MaxReale_prossimi5g_gwei2"]
    sd_diff_gwei = diff.std()

    worst_over = (t["MaxReale_prossimi5g_gwei2"] / t["FeeMax_gwei2"]).max()

    avg_cost_eur = None
    if eth_eur:
        avg_cost_eur = (GAS_LIMIT * t["FeeMax_gwei2"] * 1e-9 * eth_eur).mean()

    return {
        "PScore": p,
        "SD_ln_ratio": sd_ln,
        "SD_diff_gwei": sd_diff_gwei,
        "WorstOverrun": worst_over,
        "CostoMedio_FeeMax_EUR": avg_cost_eur,
    }


scenari = [
    ("Base", 2.33, 0.25, 0.40, 1.00),
    ("Più conservativo", 2.58, 0.25, 0.40, 1.00),
    ("Meno conservativo", 2.05, 0.25, 0.40, 1.00),
    ("Momentum alto", 2.33, 0.25, 0.60, 1.00),
    ("Momentum basso", 2.33, 0.25, 0.20, 1.00),
    ("VolAdj disattivato", 2.33, 0.00, 0.40, 1.00),
    ("VolAdj alto", 2.33, 0.50, 0.40, 1.00),
]

rows = []
for name, a0, asens, b, g in scenari:
    m = compute_metrics(a0, asens, b, g)
    rows.append(
        {
            "Scenario": name,
            "alpha_base": a0,
            "alpha_sens": asens,
            "beta": b,
            "gamma": g,
            "PScore": m["PScore"],
            "SD ln(FeeMax/MaxReale)": m["SD_ln_ratio"],
            "SD (FeeMax - MaxReale) gwei": m["SD_diff_gwei"],
            "WorstOverrun (MaxReale/FeeMax)": m["WorstOverrun"],
            "Costo medio FeeMax (EUR)": m["CostoMedio_FeeMax_EUR"],
        }
    )

sens = pd.DataFrame(rows)

# formattazione italiana
sens_fmt = sens.copy()
sens_fmt["PScore"] = sens_fmt["PScore"].apply(lambda x: fmt_pct_it(x, 2))
sens_fmt["SD ln(FeeMax/MaxReale)"] = sens_fmt["SD ln(FeeMax/MaxReale)"].apply(
    lambda x: fmt_num_it(x, 4)
)
sens_fmt["SD (FeeMax - MaxReale) gwei"] = sens_fmt[
    "SD (FeeMax - MaxReale) gwei"
].apply(lambda x: fmt_num_it(x, 2))
sens_fmt["WorstOverrun (MaxReale/FeeMax)"] = sens_fmt[
    "WorstOverrun (MaxReale/FeeMax)"
].apply(lambda x: fmt_num_it(x, 4))
if eth_eur:
    sens_fmt["Costo medio FeeMax (EUR)"] = sens_fmt["Costo medio FeeMax (EUR)"].apply(
        lambda x: fmt_num_it(x, 2) + " €"
    )

print("\n=== Sensitivity Analysis ===")
print(sens_fmt.to_string(index=False))

# -------------------------
# 10) RIEPILOGO "RATE DI SUCCESSO" (scenario base)
# -------------------------
print("\n=== Riepilogo (Scenario Base) ===")
print(f"Rate di successo / copertura (PScore): {fmt_pct_it(pscore,2)}")
print(f"Deviazione standard ln(FeeMax/MaxReale): {fmt_num_it(ln_ratio.std(),4)}")
if live_basefee_gwei is not None and eth_eur:
    live_cost_eur = GAS_LIMIT * live_basefee_gwei * 1e-9 * eth_eur
    print(
        f"Stima costo LIVE (base fee ultimo blocco) per tx gas={GAS_LIMIT}: {fmt_num_it(live_cost_eur,2)} €"
    )

print("\nGrafici salvati: fee_max_vs_reali_gwei.png" + (" e fee_max_vs_reali_eur.png" if eth_eur else ""))
