# Gas Fee Analysis

This repository contains a self-contained Ethereum gas fee analysis workflow built only on public market and blockchain data. It is intended as a portfolio project: no proprietary datasets, client information, internal references, or organization-specific logic are required to run it.

## What the analysis does

The main script estimates a conservative transaction fee threshold called `FeeMax` and compares it with the realized maximum gas price observed over the next 5 days.

The analysis covers:
- long-term trend estimation with exponential moving averages
- short-term volatility measured on 5-day log returns
- momentum and regime adjustment factors
- parameter calibration through grid search
- coverage and overrun diagnostics
- HTML report generation with charts, summary tables, and live data checks

## Public data sources

The project only uses external public endpoints:
- [Etherscan](https://etherscan.io/chart/gasprice?output=csv) for historical daily Ethereum gas price data
- [CoinGecko](https://www.coingecko.com/) for the live ETH/EUR exchange rate
- public Ethereum JSON-RPC endpoints for the latest base fee

No API keys are required.

## Repository scope

The repository is intentionally limited to the code and documentation needed to generate the analysis. Generated reports, charts, spreadsheets, office documents, and local working artifacts are excluded from version control.

## Setup

Install the required packages once:

```bash
python3 -m pip install numpy pandas matplotlib requests
```

## Generate the HTML report

Run the main script:

```bash
python3 feemax_realdata.py --last-days 10
```

Useful options:
- `--headless` generates the report without opening plot windows
- `--plot-eur` adds EUR-denominated cost charts
- `--last-days N` adds a short-term detail view for the last `N` days
- `--save-plots` saves PNG charts locally in addition to the HTML report

Examples:

```bash
python3 feemax_realdata.py --last-days 10
python3 feemax_realdata.py --last-days 10 --plot-eur
python3 feemax_realdata.py --last-days 10 --headless --save-plots
```

Each run creates a timestamped HTML file in:

```text
reports/report_YYYYMMDD_HHMMSS.html
```

On macOS you can open the generated report with:

```bash
open reports/report_YYYYMMDD_HHMMSS.html
```

## Report contents

The HTML report includes:
- live data availability checks
- calibrated model parameters
- backtest metrics against the realized 5-day forward maximum
- sensitivity analysis across selected scenarios
- failure-case diagnostics
- embedded charts in gwei and, optionally, in EUR

Core metrics include:
- `PScore`: share of days where `FeeMax` covers the realized 5-day maximum
- `OverrunRate`: share of uncovered days
- `Calibration`: distance from the target coverage level
- `BiasScore`: mean log ratio between the estimate and the realized maximum
- `Headroom`: average and median conservatism of the estimate
- `WorstOverrun`: largest realized miss versus the estimate

## Model summary

The model works in log space:

```text
ln(FeeMax) = mu_ln_LONG
           + alpha_t * sigma_5d
           + beta * max(0, MOMENTUM)
           + gamma * REGIME_ADJ
```

Where:

```text
mu_ln_LONG = 0.6 * EMA90 + 0.4 * EMA750
log_ret_5d = ln_fee(t) - ln_fee(t-5)
sigma_5d   = EWMA_std(log_ret_5d, halflife=180)
MOMENTUM   = zscore(0.6 * ROC20 + 0.4 * Accel) over 500d
REGIME_ADJ = function(exp(EMA90 - EMA750))
```

The final estimate can be capped against a rolling 30-day mean to keep the output conservative without becoming unreasonably large.

## Files

- `feemax_realdata.py`: main script that fetches data, calibrates the model, and generates the HTML report
- `gas_fee_analysis.py`: older standalone variant kept as a simpler reference implementation

## Privacy and sensitivity

This repository is intended to remain generic and non-attributable:
- only public external data sources are used
- no business identifiers or internal project names are included
- no sensitive or confidential material should be committed
