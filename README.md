# Non-Homogeneous Poisson Process and Gamma Model for Power-Outage Reliability

This project models electrical-power outage risk as a two-stage stochastic system:

1. a **non-homogeneous Poisson process (NHPP)** for the arrival of outage events over time; and
2. a **Gamma distribution** for the strictly positive restoration duration of each event.

The resulting model quantifies time-varying outage frequency, restoration-time uncertainty, expected aggregate downtime, availability, and the contrast between normal and high-disturbance conditions. It is designed as an auditable end-to-end analysis pipeline rather than a black-box predictor.

## Purpose

The project answers two operational reliability questions that should be separated statistically:

- *When do outages occur, and how does their rate vary over time?*
- *Once an outage occurs, how long is restoration likely to take?*

Conflating these mechanisms would obscure whether poor reliability is driven by a higher event rate, longer restoration, or both. The NHPP--Gamma formulation retains that distinction and supports Monte Carlo estimates of system-level downtime and availability.

## Dataset

The pipeline uses the public Kaggle dataset [`autunno/15-years-of-power-outages`](https://www.kaggle.com/datasets/autunno/15-years-of-power-outages). The version used for the reported run contained U.S. grid-disruption records from 2000 onward and supplied:

- event start date/time;
- restoration date/time;
- event description and tags; and
- number of customers affected, when available.

The raw dataset and all regenerated analysis outputs are intentionally excluded from the repository. They may contain record-level information and can be re-created locally by running the pipeline.

## Model

### 1. NHPP outage-arrival model

Let \(N(t)\) denote the number of outage starts by time \(t\). The model assumes

\[
N(t + \Delta t) - N(t) \sim \operatorname{Poisson}\!\left(\int_t^{t+\Delta t}\lambda(u)\,du\right),
\]

where \(\lambda(t)\) is a time-varying outage intensity. The script aggregates valid outage starts into daily counts \(X_i\), estimates an empirical rate \(X_i/\Delta t\), then uses a centered 30-day rolling mean to obtain the simulated expected count \(\hat\lambda(t_i)\Delta t\). A Poisson draw is generated independently for every interval.

This is a non-parametric intensity smoother, not a covariate-driven or seasonal NHPP. It is appropriate for describing the observed temporal pattern, but it is not an out-of-sample forecasting model: random train/test splits would break temporal dependence and are not used here.

### 2. Gamma restoration-time model

For a positive restoration time \(T_r\) in hours,

\[
T_r \sim \operatorname{Gamma}(k,\theta), \qquad
\mathbb{E}[T_r] = k\theta, \quad
\operatorname{Var}(T_r) = k\theta^2.
\]

The pipeline first calculates method-of-moments estimates, then fits \(k\) and \(\theta\) by maximum likelihood with the location fixed at zero. Only records with a valid positive duration are used. Fixing `loc=0` respects the support of a duration variable and makes the fit reproducible.

### 3. Scenario and reliability analysis

Events whose description or tags match severe-weather and natural-disturbance terms (for example, `storm`, `wind`, `flood`, `wildfire`, or `earthquake`) form the **High-disturbance** scenario; all other events form **Normal**. Each scenario receives its own smoothed NHPP intensity and Gamma duration fit.

For a horizon \(T\), expected aggregate downtime is approximated by

\[
\mathbb{E}[D(T)] \approx \mathbb{E}[N(T)]\,\mathbb{E}[T_r],
\]

and the reported availability approximation is

\[
A = 1 - \frac{D(T)}{T}.
\]

The script also runs 500 NHPP--Gamma Monte Carlo replications. SAIDI and SAIFI are reported only as proxy indicators because the source has affected customers per event but no true total number of customers served; the maximum observed affected-customer count is used as a denominator proxy.

## Results from the included analysis run

The following values come from the generated `model_parameters.json`, `reliability_summary.csv`, and `scenario_comparison.csv` produced by the current configuration (`seed = 42`, daily aggregation, 30-day window, 500 simulations).

| Quantity | Result |
| --- | ---: |
| Valid outage starts | 1,645 |
| Positive restoration durations used by Gamma fit | 1,507 |
| Observation horizon | 126,552 h |
| Mean observed outage rate | 0.3120 outages/day |
| NHPP expected total \(m(T)\) | 1,643.97 outages |
| NHPP one-path count MAE / RMSE | 0.4470 / 1.0044 daily outages |
| Gamma shape \(k\) / scale \(\theta\) | 0.3938 / 123.9563 h |
| Mean observed restoration time | 48.82 h |
| Gamma KS statistic / p-value | 0.0609 / \(2.65\times10^{-5}\) |
| Observed aggregate downtime | 73,566.23 h |
| Expected aggregate downtime | 80,252.80 h |
| Observed availability approximation | 0.4187 |
| Monte Carlo mean availability | 0.3658 |
| Monte Carlo 5th--95th availability percentile | 0.3169--0.4138 |

Scenario-specific results show why separating disturbance regimes matters:

| Scenario | Events | Rate (outages/day) | Mean restoration (h) | Expected outages | Expected downtime (h) | Availability approximation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 749 | 0.1420 | 27.78 | 749.11 | 20,809.83 | 0.8611 |
| High-disturbance | 896 | 0.1699 | 64.05 | 894.86 | 57,318.24 | 0.5576 |

The high-disturbance subset has both a higher arrival rate and substantially longer mean restoration time. Its Gamma fit is also materially more plausible under the one-sample KS diagnostic (KS p = 0.4042) than the pooled fit. The pooled Gamma fit is formally rejected by the KS test, so its distributional summaries should be treated as a compact operational approximation rather than a validated universal duration law. The current NHPP diagnostic compares a single simulated path with in-sample daily counts; its low correlation (0.1590) reinforces that the smoothed rate captures broad intensity rather than individual-day realizations.

The reported availability is an event-time approximation, not feeder-level system availability. Summed event durations can overlap across regions and customers, and the source does not supply network topology or a population denominator. It must therefore not be used as a regulatory SAIDI/SAIFI or system-availability estimate.

## Repository structure

```text
.
├── power_outage_nhpp_gamma_model.py  # End-to-end modelling, diagnostics, plots, and simulation
├── requirements.txt                  # Python dependencies
├── README.md
└── .gitignore                        # Excludes datasets, outputs, credentials, environments, and caches
```

When the script runs, it creates `data/` for the Kaggle download/cache and `outputs/` for cleaned tables, model metadata, simulations, and PNG/SVG diagnostics. Both directories are ignored by Git.

## Requirements

- Python 3.11 or later
- A Kaggle account and API credentials if the source dataset is not already cached locally
- Packages listed in `requirements.txt`:
  - `kaggle`, `kagglehub`
  - `pandas`, `numpy`, `scipy`, `matplotlib`
  - `openpyxl`, `xlrd` for Excel-compatible input discovery

Keep Kaggle credentials outside the repository. A standard setup places `kaggle.json` in `~/.kaggle/`; alternatively, provide a local input file through `OUTAGE_DATA_PATH`.

## Run

Create and activate an isolated environment, then install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run with the Kaggle dataset download/cache path:

```bash
python power_outage_nhpp_gamma_model.py
```

Or run against a local CSV, TSV, XLSX, or XLS file without downloading data:

```bash
OUTAGE_DATA_PATH=/absolute/path/to/outage_records.csv python power_outage_nhpp_gamma_model.py
```

The pipeline automatically discovers a compatible tabular file and infers likely start-time, restoration-time, duration, cause, and affected-customer columns. Inspect `outputs/model_parameters.json` after every run to confirm the inferred mapping before interpreting results.

## Generated outputs

`outputs/` is regenerated each time and includes:

- `cleaned_outage_data.csv` — standardized records with timestamps, duration, cause text, and scenario;
- `outage_count_timeseries.csv` — daily counts and empirical/smoothed intensities;
- `reliability_summary.csv` and `scenario_comparison.csv` — reliability metrics;
- `monte_carlo_results.csv` — 500 simulation outcomes;
- `model_parameters.json` — data mapping, fitted parameters, diagnostics, and random seed; and
- plots for outage counts, intensity, cumulative counts, Gamma diagnostics, scenarios, and simulated downtime.

## Reproducibility and limitations

- The random seed is fixed at `42`; change `RANDOM_SEED` to explore simulation variation.
- `AGGREGATION_FREQUENCY = "D"` and `ROLLING_WINDOW_INTERVALS = 30` are modelling choices, not facts supplied by the data. Sensitivity analysis over interval size and smoothing width is recommended.
- The model treats arrivals and restoration durations as conditionally separate. A marked or covariate-dependent point process would be preferable if weather severity, geography, seasonality, or grid state are available.
- Missing restoration timestamps reduce the Gamma fitting sample but do not remove events from the NHPP arrival process. If missingness is systematic, duration estimates may be biased.
- The severe-disturbance classifier is keyword based. It is transparent but susceptible to label noise and should be replaced with validated event taxonomy or external hazard labels for production use.
