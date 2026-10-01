"""
End-to-end NHPP and Gamma modelling pipeline for electrical power outages.

The script downloads a power-outage dataset with the Kaggle API when needed,
inspects candidate tabular files, infers key columns, models outage arrivals
with a Non-Homogeneous Poisson Process, models restoration duration with a
Gamma distribution, compares normal and high-disturbance scenarios, evaluates
the models, and saves tables plus plots under outputs/.

How the script is organized:
1. Configuration and constants define paths, model frequency, and simulation size.
2. Data-loading helpers find the dataset and choose the most relevant table.
3. Column-inference helpers map flexible real-world column names to model fields.
4. Preprocessing converts timestamps, durations, causes, and customer counts.
5. NHPP functions model outage arrivals as time-varying Poisson counts.
6. Gamma functions model positive restoration durations in hours.
7. Reliability, scenario, Monte Carlo, plot, and output functions complete the
   analysis pipeline.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    KaggleApi = None

try:
    import kagglehub
except ImportError:
    kagglehub = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Kaggle dataset requested in the assignment prompt.
DATASET_SLUG = "autunno/15-years-of-power-outages"

# Local cache used by the Kaggle API downloader. If tabular files already exist
# here, the script reuses them instead of downloading the dataset again.
KAGGLE_DATASET_DIR = Path("data") / "kaggle" / DATASET_SLUG.replace("/", "_")

# All generated CSV, JSON, and PNG files are written under this directory.
OUTPUT_DIR = Path("outputs")

# This controls the NHPP time step. "D" means each X_i is the number of
# outages observed on one day. Change to "h", "W", or "ME" if needed.
AGGREGATION_FREQUENCY = "D"  # Change to "h", "W", or "ME" for hourly/weekly/monthly.

# The rolling mean smooths noisy empirical counts into a stable lambda(t).
ROLLING_WINDOW_INTERVALS = 30

# Monte Carlo repetitions used to estimate uncertainty in downtime/availability.
N_MONTE_CARLO_SIMULATIONS = 500
RANDOM_SEED = 42

# Optional manual dataset override. Example:
# OUTAGE_DATA_PATH="/path/to/file.csv" python power_outage_nhpp_gamma_model.py
LOCAL_DATA_ENV_VAR = "OUTAGE_DATA_PATH"
SUPPORTED_TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}

# Files created by this script. These are excluded from automatic fallback
# dataset discovery so the pipeline does not accidentally read its own outputs
# as if they were raw outage records.
GENERATED_OUTPUT_FILENAMES = {
    "cleaned_outage_data.csv",
    "outage_count_timeseries.csv",
    "reliability_summary.csv",
    "scenario_comparison.csv",
    "monte_carlo_results.csv",
}

# Terms used to label records as high-disturbance when cause/event text exists.
HIGH_DISTURBANCE_KEYWORDS = (
    "severe weather",
    "storm",
    "thunderstorm",
    "hurricane",
    "wildfire",
    "fire",
    "winter storm",
    "ice",
    "wind",
    "flood",
    "flooding",
    "lightning",
    "earthquake",
    "tornado",
    "natural disaster",
    "heat",
    "cold",
    "weather",
    "snow",
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_SEED)
warnings.filterwarnings("default", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Lightweight result containers
# ---------------------------------------------------------------------------

@dataclass
class ColumnMapping:
    """
    Inferred dataset fields used by the modelling pipeline.

    Most outage datasets do not use identical column names. This container keeps
    the guessed column names in one object so preprocessing can stay readable.
    Each field is optional because some datasets may omit duration, cause, or
    customer-impact information.
    """

    start_timestamp_col: str | None
    start_date_col: str | None
    start_time_col: str | None
    restoration_timestamp_col: str | None
    restoration_date_col: str | None
    restoration_time_col: str | None
    duration_col: str | None
    cause_col: str | None
    cause_columns: list[str]
    customers_col: str | None


@dataclass
class GammaFit:
    """
    Estimated Gamma restoration-time distribution parameters and diagnostics.

    k is the Gamma shape parameter and theta is the scale parameter. The object
    stores both method-of-moments and maximum-likelihood estimates so the final
    JSON output is auditable.
    """

    method: str
    k: float
    theta: float
    loc: float
    mom_k: float
    mom_theta: float
    mle_k: float | None
    mle_theta: float | None
    n: int
    empirical_mean: float
    empirical_variance: float
    fitted_mean: float
    fitted_variance: float
    log_likelihood: float
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float


# ---------------------------------------------------------------------------
# General utility helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    """Print a readable terminal section header."""

    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def normalize_column_name(name: str) -> str:
    """Normalize a column name for fuzzy matching.

    The scoring functions compare lowercase words instead of exact column
    strings, so names such as "Date Event Began" and "date_event_began" can be
    handled by the same logic.
    """

    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def compact_column_name(name: str) -> str:
    """Normalize a column name to compact alphanumeric form."""

    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def as_serializable(value: Any) -> Any:
    """Convert numpy, pandas, and pathlib objects into JSON-safe objects.

    Pandas and NumPy use types that json.dump cannot serialize directly. This
    helper recursively converts those values to built-in Python types before the
    model parameter file is written.
    """

    if isinstance(value, dict):
        return {str(k): as_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return [as_serializable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (Path,)):
        return str(value)
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass
    return value


# ---------------------------------------------------------------------------
# Timestamp, numeric, and duration parsing
# ---------------------------------------------------------------------------

def safe_to_datetime(series: pd.Series) -> pd.Series:
    """Parse a pandas Series to datetimes while tolerating mixed string formats.

    Real outage data often mixes blank strings, "Unknown", dates, and full
    timestamps. Invalid values are kept as NaT so they can be filtered later.
    """

    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.str.lower().isin({"", "nan", "none", "null", "unknown", "unk"}))
    try:
        # Pandas 2+ supports format="mixed", which is useful for inconsistent
        # date formats in public datasets. The except block keeps compatibility.
        return pd.to_datetime(cleaned, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(cleaned, errors="coerce")


def combine_date_time(
    df: pd.DataFrame,
    date_col: str | None,
    time_col: str | None,
    timestamp_col: str | None = None,
) -> pd.Series:
    """Build a timestamp from one datetime column or separate date/time columns.

    The function prefers a single full timestamp column when available. If the
    dataset stores dates and clock times separately, it combines them before
    parsing.
    """

    if timestamp_col and timestamp_col in df.columns:
        return safe_to_datetime(df[timestamp_col])

    if not date_col or date_col not in df.columns:
        # Return an all-missing Series so downstream code can apply one uniform
        # missing-timestamp filter instead of handling None separately.
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    date_text = df[date_col].astype("string").str.strip()
    date_text = date_text.mask(date_text.str.lower().isin({"", "nan", "none", "null", "unknown", "unk"}))

    if time_col and time_col in df.columns:
        time_text = df[time_col].astype("string").str.strip()
        time_text = time_text.mask(time_text.str.lower().isin({"", "nan", "none", "null", "unknown", "unk"}))
        combined = date_text.fillna("") + " " + time_text.fillna("")
        combined = combined.str.strip().mask(date_text.isna())
        return safe_to_datetime(combined)

    return safe_to_datetime(date_text)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Convert strings such as '420,000' or 'Unknown' into numeric values.

    This is used for customer counts and other optional numeric fields. Only the
    first numeric token is kept, which is safer than assuming the full text is a
    clean number.
    """

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype("string").str.strip()
    text = text.mask(text.str.lower().isin({"", "nan", "none", "null", "unknown", "unk", "n/a", "na"}))
    text = text.str.replace(",", "", regex=False)
    extracted = text.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def parse_duration_value_to_hours(value: Any, default_unit: str) -> float:
    """Parse one duration value into hours.

    All restoration durations are normalized to hours so downtime, Gamma
    parameters, and availability use the same unit.
    """

    if pd.isna(value):
        return np.nan
    if isinstance(value, pd.Timedelta):
        return value.total_seconds() / 3600
    if isinstance(value, np.timedelta64):
        return pd.to_timedelta(value).total_seconds() / 3600
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return convert_numeric_duration_to_hours(numeric, default_unit)

    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null", "unknown", "unk", "n/a", "na"}:
        return np.nan

    text = text.replace(",", "")
    # Handle textual durations such as "2 days 4 hours" or "90 minutes".
    unit_matches = re.findall(
        r"([-+]?\d*\.?\d+)\s*(days?|d\b|hours?|hrs?|hr\b|h\b|minutes?|mins?|min\b|m\b|seconds?|secs?|sec\b|s\b)",
        text,
    )
    if unit_matches:
        total_hours = 0.0
        for number_text, unit in unit_matches:
            amount = float(number_text)
            if unit.startswith("d"):
                total_hours += amount * 24
            elif unit.startswith(("h", "hr")):
                total_hours += amount
            elif unit.startswith(("m", "min")):
                total_hours += amount / 60
            elif unit.startswith(("s", "sec")):
                total_hours += amount / 3600
        return total_hours

    numeric_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if numeric_match:
        return convert_numeric_duration_to_hours(float(numeric_match.group(0)), default_unit)

    return np.nan


def infer_duration_unit_from_name(name: str | None) -> str:
    """Infer the unit for numeric duration values from the column name.

    If no unit appears in the name, hours is the conservative default because
    restoration duration is often reported in hours.
    """

    normalized = normalize_column_name(name or "")
    if any(term in normalized for term in ("minute", "minutes", "min", "mins")):
        return "minutes"
    if any(term in normalized for term in ("day", "days")):
        return "days"
    if any(term in normalized for term in ("second", "seconds", "sec", "secs")):
        return "seconds"
    return "hours"


def convert_numeric_duration_to_hours(value: float, unit: str) -> float:
    """Convert numeric duration from the inferred unit into hours."""

    if unit == "minutes":
        return value / 60
    if unit == "days":
        return value * 24
    if unit == "seconds":
        return value / 3600
    return value


def parse_duration_series(series: pd.Series, column_name: str | None) -> pd.Series:
    """Convert a duration column to hours with unit inference."""

    default_unit = infer_duration_unit_from_name(column_name)
    parsed = series.map(lambda value: parse_duration_value_to_hours(value, default_unit))
    parsed = pd.to_numeric(parsed, errors="coerce")

    # If the name gave no unit and the data strongly looks like minutes, convert.
    if default_unit == "hours":
        positive = parsed[(parsed > 0) & np.isfinite(parsed)]
        if len(positive) > 0 and positive.median() > 24 * 30:
            warnings.warn(
                "Duration column has no unit in its name and very large values; "
                "treating numeric durations as minutes."
            )
            parsed = parsed / 60
    return parsed


# ---------------------------------------------------------------------------
# Dataset discovery, loading, and inspection
# ---------------------------------------------------------------------------

def discover_tabular_files(path: Path) -> list[Path]:
    """Find supported tabular files under a file or directory path.

    Kaggle datasets often contain more than one file. Returning every supported
    tabular file lets the script inspect and score them before loading the final
    modelling table.
    """

    if path.is_file() and path.suffix.lower() in SUPPORTED_TABULAR_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(
            file
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in SUPPORTED_TABULAR_EXTENSIONS
        )
    return []


def is_generated_output_file(path: Path) -> bool:
    """Return True when a file is one of this script's generated CSV outputs."""

    return path.name in GENERATED_OUTPUT_FILENAMES or OUTPUT_DIR.name in path.parts


def score_fallback_dataset_file(path: Path) -> int:
    """Score local fallback files using both filename and sampled columns.

    Column evidence receives much more weight than filename evidence because
    generated files can have names such as "outage_count_timeseries.csv" while
    lacking the raw event-start/restoration columns required for modelling.
    """

    name = normalize_column_name(path.name)
    filename_score = 0
    if "outage" in name:
        filename_score += 4
    if "grid" in name or "disruption" in name or "power" in name:
        filename_score += 2
    if "standardized" in name:
        filename_score += 1

    try:
        sample = read_tabular_file(path, nrows=50)
        column_score = score_columns_for_outage_data(list(sample.columns))
    except Exception:
        # Some local files may be unrelated or require optional readers such as
        # openpyxl. During fallback discovery they should not interrupt or
        # clutter the run; unreadable candidates simply receive no column score.
        column_score = 0

    return column_score * 10 + filename_score


def download_dataset_with_kaggle_api() -> Path | None:
    """Download the Kaggle dataset with the official Kaggle API if needed.

    Returns the local dataset directory when the download/cache is usable. If
    the Kaggle package is missing, credentials are not configured, or the API
    request fails, the caller receives None and can try another data source.
    """

    cached_files = discover_tabular_files(KAGGLE_DATASET_DIR)
    if cached_files:
        print(f"Using cached Kaggle API dataset directory: {KAGGLE_DATASET_DIR.resolve()}")
        return KAGGLE_DATASET_DIR

    if KaggleApi is None:
        warnings.warn(
            "The kaggle package is not installed. Install it with 'pip install kaggle' "
            "and configure ~/.kaggle/kaggle.json to enable Kaggle API download. "
            "Trying secondary/fallback data sources."
        )
        return None

    try:
        KAGGLE_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading Kaggle dataset '{DATASET_SLUG}' to {KAGGLE_DATASET_DIR.resolve()}")
        api.dataset_download_files(
            DATASET_SLUG,
            path=str(KAGGLE_DATASET_DIR),
            unzip=True,
            quiet=False,
        )
    except Exception as exc:
        warnings.warn(
            "Kaggle API download failed. Check that the kaggle package is installed "
            f"and credentials are configured. Original error: {exc}. "
            "Trying secondary/fallback data sources."
        )
        return None

    downloaded_files = discover_tabular_files(KAGGLE_DATASET_DIR)
    if not downloaded_files:
        warnings.warn(
            f"Kaggle API download completed, but no CSV/TSV/Excel files were found in {KAGGLE_DATASET_DIR}. "
            "Trying secondary/fallback data sources."
        )
        return None

    print(f"Kaggle API dataset ready: {KAGGLE_DATASET_DIR.resolve()}")
    return KAGGLE_DATASET_DIR


def download_dataset_with_kagglehub() -> Path | None:
    """Download the dataset with KaggleHub as a secondary option."""

    if kagglehub is None:
        return None

    try:
        dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))
        print(f"KaggleHub dataset path: {dataset_path}")
        return dataset_path
    except Exception as exc:
        warnings.warn(f"KaggleHub download failed: {exc}. Trying local fallback data.")
        return None


def download_or_find_dataset() -> Path:
    """Download the Kaggle dataset, or locate a local fallback dataset.

    The primary path is the official Kaggle API. The fallback paths make the
    script runnable in environments where the Kaggle package or credentials are
    unavailable.
    """

    print_section("Dataset Download / Discovery")

    kaggle_api_path = download_dataset_with_kaggle_api()
    if kaggle_api_path is not None:
        return kaggle_api_path

    kagglehub_path = download_dataset_with_kagglehub()
    if kagglehub_path is not None:
        return kagglehub_path

    env_path = Path(str(Path.cwd()))
    if LOCAL_DATA_ENV_VAR in os.environ:
        env_path = Path(os.environ[LOCAL_DATA_ENV_VAR]).expanduser()
        files = discover_tabular_files(env_path)
        if files:
            print(f"Using local dataset path from {LOCAL_DATA_ENV_VAR}: {env_path}")
            return env_path
        warnings.warn(f"{LOCAL_DATA_ENV_VAR} was set to {env_path}, but no tabular files were found.")

    search_roots = [
        Path.cwd(),
        Path.cwd() / "data",
        Path.home() / "Downloads",
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if root.exists():
            candidates.extend(discover_tabular_files(root))

    scored: list[tuple[int, Path]] = []
    for file in candidates:
        if is_generated_output_file(file):
            continue
        score = score_fallback_dataset_file(file)
        scored.append((score, file))

    scored = sorted(scored, key=lambda item: (item[0], item[1].stat().st_mtime), reverse=True)
    if scored and scored[0][0] > 0:
        fallback_file = scored[0][1]
        print(f"Using local fallback file: {fallback_file}")
        return fallback_file

    raise RuntimeError(
        "Could not download or locate the outage dataset. Install/configure the Kaggle API, "
        f"or set {LOCAL_DATA_ENV_VAR} to a CSV/Excel file or dataset directory."
    )


def read_tabular_file(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read CSV, TSV, or Excel data into a DataFrame.

    Excel files can contain multiple sheets. The script selects the sheet with
    the largest sample shape as a practical default for raw datasets.
    """

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        excel = pd.ExcelFile(path)
        best_sheet = None
        best_shape = (-1, -1)
        for sheet_name in excel.sheet_names:
            sample = pd.read_excel(path, sheet_name=sheet_name, nrows=nrows)
            shape = sample.shape
            if shape > best_shape:
                best_shape = shape
                best_sheet = sheet_name
        if best_sheet is None:
            raise ValueError(f"No readable sheets found in {path}")
        print(f"Selected sheet '{best_sheet}' from {path.name}")
        return pd.read_excel(path, sheet_name=best_sheet, nrows=nrows)
    raise ValueError(f"Unsupported tabular file extension: {path.suffix}")


def score_columns_for_outage_data(columns: list[str]) -> int:
    """Score how likely a file is to contain outage records.

    A higher score means the file has columns that look like outage starts,
    restorations, causes, durations, or customer impact. This avoids hardcoding
    one exact Kaggle filename.
    """

    normalized = [normalize_column_name(col) for col in columns]
    compacted = [compact_column_name(col) for col in columns]
    joined = " ".join(normalized)
    compact_joined = " ".join(compacted)

    score = 0
    if any("outage" in col or "event" in col for col in normalized):
        score += 2
    if any(("start" in col or "began" in col or "begin" in col) and "date" in col for col in normalized):
        score += 5
    if any(("restore" in col or "restoration" in col or "end" in col) for col in normalized):
        score += 5
    if any("duration" in col or "recovery" in col for col in normalized):
        score += 4
    if any("customer" in col and ("affected" in col or "interrupt" in col) for col in normalized):
        score += 3
    if any("cause" in col or "tag" in col or "eventdescription" in col for col in compacted):
        score += 3
    if "dateeventbegan" in compact_joined or "timeeventbegan" in compact_joined:
        score += 6
    return score


def inspect_and_load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, Path]:
    """Inspect all files and load the most relevant tabular dataset.

    The printed inspection is intentional: it makes the script transparent when
    column names differ from the expected outage schema.
    """

    print_section("Dataset File Inspection")
    files = discover_tabular_files(dataset_path)
    if not files:
        raise RuntimeError(f"No CSV/TSV/Excel files found in {dataset_path}")

    print("Discovered tabular files:")
    for file in files:
        print(f" - {file}")

    candidates: list[tuple[int, int, int, Path, list[str]]] = []
    for file in files:
        try:
            # Read only a sample first so large files can be scored cheaply.
            sample = read_tabular_file(file, nrows=100)
            score = score_columns_for_outage_data(list(sample.columns))
            candidates.append((score, sample.shape[0], sample.shape[1], file, list(sample.columns)))
            print(f"\nCandidate: {file.name}")
            print(f"  shape sample: {sample.shape}")
            print(f"  relevance score: {score}")
            print(f"  columns: {list(sample.columns)}")
        except Exception as exc:
            warnings.warn(f"Could not inspect {file}: {exc}")

    if not candidates:
        raise RuntimeError("No readable tabular files were found.")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_file = candidates[0][3]
    print(f"\nSelected dataset file: {best_file}")

    df = read_tabular_file(best_file)
    print_section("Loaded Dataset Preview")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))
    print("\nData types:")
    print(df.dtypes.astype(str))
    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))
    print("\nSample rows:")
    print(df.head(5).to_string(index=False))

    return df, best_file


# ---------------------------------------------------------------------------
# Column-name scoring and inference
# ---------------------------------------------------------------------------

def score_start_timestamp_column(name: str) -> int:
    """Score columns that may contain full event start timestamps.

    Positive terms add evidence that the column is a start time; restoration or
    end terms subtract evidence so the start and end fields do not get swapped.
    """

    normalized = normalize_column_name(name)
    score = 0
    if any(term in normalized for term in ("timestamp", "datetime", "date time")):
        score += 4
    if any(term in normalized for term in ("start", "began", "begin", "occurrence", "occurred", "event")):
        score += 3
    if "date" in normalized and "time" in normalized:
        score += 2
    if any(term in normalized for term in ("restore", "restoration", "end", "resolved", "repair")):
        score -= 8
    return score


def score_start_date_column(name: str) -> int:
    """Score columns that may contain event start dates."""

    normalized = normalize_column_name(name)
    score = 0
    if "date" in normalized:
        score += 3
    if any(term in normalized for term in ("start", "began", "begin", "event", "outage", "occurred")):
        score += 3
    if "date event began" in normalized:
        score += 5
    if "time" in normalized and "date" not in normalized:
        score -= 2
    if any(term in normalized for term in ("restore", "restoration", "end", "resolved", "repair")):
        score -= 8
    return score


def score_start_time_column(name: str) -> int:
    """Score columns that may contain event start clock times."""

    normalized = normalize_column_name(name)
    score = 0
    if "time" in normalized:
        score += 3
    if any(term in normalized for term in ("start", "began", "begin", "event", "outage", "occurred")):
        score += 3
    if "time event began" in normalized:
        score += 5
    if "date" in normalized and "time" not in normalized:
        score -= 2
    if any(term in normalized for term in ("restore", "restoration", "end", "resolved", "repair")):
        score -= 8
    return score


def score_restoration_timestamp_column(name: str) -> int:
    """Score columns that may contain full restoration timestamps.

    Restoration columns define the end of an outage and are used to compute
    recovery duration when a direct duration column is absent.
    """

    normalized = normalize_column_name(name)
    score = 0
    if any(term in normalized for term in ("timestamp", "datetime", "date time")):
        score += 3
    if any(term in normalized for term in ("restore", "restoration", "end", "resolved", "repair")):
        score += 5
    if "date" in normalized and "time" in normalized:
        score += 2
    if any(term in normalized for term in ("start", "began", "begin")):
        score -= 8
    return score


def score_restoration_date_column(name: str) -> int:
    """Score columns that may contain restoration dates."""

    normalized = normalize_column_name(name)
    score = 0
    if "date" in normalized:
        score += 3
    if any(term in normalized for term in ("restore", "restoration", "end", "resolved", "repair")):
        score += 5
    if "date of restoration" in normalized:
        score += 5
    if any(term in normalized for term in ("start", "began", "begin")):
        score -= 8
    return score


def score_restoration_time_column(name: str) -> int:
    """Score columns that may contain restoration clock times."""

    normalized = normalize_column_name(name)
    score = 0
    if "time" in normalized:
        score += 3
    if any(term in normalized for term in ("restore", "restoration", "end", "resolved", "repair")):
        score += 5
    if "time of restoration" in normalized:
        score += 5
    if any(term in normalized for term in ("start", "began", "begin")):
        score -= 8
    return score


def score_duration_column(name: str) -> int:
    """Score columns that may contain restoration duration values.

    Date/time restoration columns are penalized here because they should be used
    to calculate duration, not mistaken for duration itself.
    """

    normalized = normalize_column_name(name)
    score = 0
    if "duration" in normalized:
        score += 8
    if "recovery" in normalized:
        score += 5
    if "restoration" in normalized and any(term in normalized for term in ("hour", "minute", "duration")):
        score += 4
    if any(term in normalized for term in ("hours", "hour", "minutes", "minute", "mins", "days", "seconds")):
        score += 2
    if "date of restoration" in normalized or "time of restoration" in normalized:
        score -= 10
    if normalized.startswith("time ") and "duration" not in normalized:
        score -= 6
    return score


def score_cause_column(name: str) -> int:
    """Score columns that may contain outage cause or event category."""

    normalized = normalize_column_name(name)
    compacted = compact_column_name(name)
    score = 0
    if "cause" in normalized:
        score += 6
    if "event description" in normalized or "event type" in normalized:
        score += 6
    if "tags" in normalized or "tag" == normalized:
        score += 5
    if any(term in normalized for term in ("category", "disturbance", "incident", "hazard")):
        score += 3
    if compacted == "eventdescription":
        score += 5
    return score


def score_customers_column(name: str) -> int:
    """Score columns that may contain affected customer counts."""

    normalized = normalize_column_name(name)
    score = 0
    if "customer" in normalized:
        score += 5
    if any(term in normalized for term in ("affected", "interrupted", "interruptions", "lost", "served")):
        score += 4
    if "number" in normalized or "count" in normalized:
        score += 1
    return score


def best_column(columns: list[str], scorer, min_score: int) -> str | None:
    """Return the highest-scoring column, or None if no score clears the threshold."""

    scored = [(scorer(col), col) for col in columns]
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored and scored[0][0] >= min_score:
        return scored[0][1]
    return None


def infer_columns(df: pd.DataFrame) -> ColumnMapping:
    """Infer key timestamp, duration, cause, and customer fields.

    The model only proceeds automatically when it can identify a start timestamp
    and a duration source. Optional fields such as cause and customers affected
    are used when available.
    """

    columns = list(df.columns)
    # Use every likely cause-like column together; for this dataset that means
    # both "Event Description" and "Tags" help classify disturbance scenarios.
    cause_columns = [col for col in columns if score_cause_column(col) >= 5]

    # Each best_column call uses a different scorer so one column can be judged
    # according to the semantics needed for that specific modelling field.
    mapping = ColumnMapping(
        start_timestamp_col=best_column(columns, score_start_timestamp_column, 7),
        start_date_col=best_column(columns, score_start_date_column, 4),
        start_time_col=best_column(columns, score_start_time_column, 4),
        restoration_timestamp_col=best_column(columns, score_restoration_timestamp_column, 7),
        restoration_date_col=best_column(columns, score_restoration_date_column, 5),
        restoration_time_col=best_column(columns, score_restoration_time_column, 5),
        duration_col=best_column(columns, score_duration_column, 5),
        cause_col=best_column(columns, score_cause_column, 5),
        cause_columns=cause_columns,
        customers_col=best_column(columns, score_customers_column, 6),
    )

    print_section("Inferred Column Mapping")
    print(json.dumps(as_serializable(mapping.__dict__), indent=2))

    if not mapping.start_timestamp_col and not mapping.start_date_col:
        raise ValueError(
            "Could not infer an outage start timestamp/date column. "
            f"Available columns are: {list(df.columns)}. "
            "Map a column containing outage start time/date manually in infer_columns()."
        )

    has_duration_source = bool(mapping.duration_col) or bool(mapping.restoration_timestamp_col) or bool(
        mapping.restoration_date_col
    )
    if not has_duration_source:
        raise ValueError(
            "Could not infer a duration/restoration field. "
            f"Available columns are: {list(df.columns)}. "
            "Expected a duration column or restoration date/time columns."
        )

    return mapping


def preprocess_data(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
    """Clean raw outage records and create modelling columns.

    The key outputs are:
    - start_timestamp: outage arrival time for NHPP counting.
    - duration_hours: positive restoration duration for Gamma fitting.
    - cause_text: normalized event/cause text for scenario classification.
    - customers_affected: optional numeric field for SAIDI/SAIFI proxies.
    """

    print_section("Preprocessing")

    cleaned = df.copy()
    initial_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    duplicate_rows = initial_rows - len(cleaned)

    # Arrival time drives N(t); it must not be confused with restoration time.
    cleaned["start_timestamp"] = combine_date_time(
        cleaned,
        mapping.start_date_col,
        mapping.start_time_col,
        mapping.start_timestamp_col,
    )

    restoration_timestamp = combine_date_time(
        cleaned,
        mapping.restoration_date_col,
        mapping.restoration_time_col,
        mapping.restoration_timestamp_col,
    )
    cleaned["restoration_timestamp"] = restoration_timestamp

    if mapping.duration_col and mapping.duration_col in cleaned.columns:
        # Prefer an explicit duration column when the dataset provides one.
        cleaned["duration_hours"] = parse_duration_series(cleaned[mapping.duration_col], mapping.duration_col)
        duration_source = f"duration column '{mapping.duration_col}'"
    else:
        # Otherwise derive restoration duration from end minus start timestamps.
        cleaned["duration_hours"] = (
            cleaned["restoration_timestamp"] - cleaned["start_timestamp"]
        ).dt.total_seconds() / 3600
        duration_source = "restoration timestamp minus start timestamp"

    # If a duration column was inferred but produced too few usable values, use timestamps.
    timestamp_duration = (
        cleaned["restoration_timestamp"] - cleaned["start_timestamp"]
    ).dt.total_seconds() / 3600
    usable_from_duration = int(((cleaned["duration_hours"] > 0) & cleaned["duration_hours"].notna()).sum())
    usable_from_timestamps = int(((timestamp_duration > 0) & timestamp_duration.notna()).sum())
    if mapping.duration_col and usable_from_timestamps > usable_from_duration:
        # If the explicit duration column is mostly unusable but timestamps work,
        # the timestamp-derived duration is the safer modelling input.
        cleaned["duration_hours"] = timestamp_duration
        duration_source = "restoration timestamp minus start timestamp"

    # Gamma duration must be strictly positive; invalid values are kept as NaN
    # so they are excluded from duration fitting without removing arrival events.
    cleaned.loc[~np.isfinite(cleaned["duration_hours"]), "duration_hours"] = np.nan
    cleaned.loc[cleaned["duration_hours"] <= 0, "duration_hours"] = np.nan

    if mapping.customers_col and mapping.customers_col in cleaned.columns:
        cleaned["customers_affected"] = clean_numeric_series(cleaned[mapping.customers_col])
        cleaned.loc[cleaned["customers_affected"] < 0, "customers_affected"] = np.nan
    else:
        cleaned["customers_affected"] = np.nan

    if mapping.cause_columns:
        cause_text = cleaned[mapping.cause_columns].astype("string").fillna("").agg(" | ".join, axis=1)
    elif mapping.cause_col and mapping.cause_col in cleaned.columns:
        cause_text = cleaned[mapping.cause_col].astype("string").fillna("")
    else:
        cause_text = pd.Series("", index=cleaned.index, dtype="string")
    cleaned["cause_text"] = cause_text.str.lower()

    # A row without an outage start cannot contribute to arrival counts. A row
    # with no duration can still contribute to N(t), so it is not dropped here.
    valid_start = cleaned["start_timestamp"].notna()
    valid_duration = cleaned["duration_hours"].notna() & (cleaned["duration_hours"] > 0)
    cleaned = cleaned.loc[valid_start].sort_values("start_timestamp").reset_index(drop=True)

    print(f"Initial rows: {initial_rows}")
    print(f"Duplicate rows removed: {duplicate_rows}")
    print(f"Rows with valid start timestamp: {len(cleaned)}")
    print(f"Rows with positive duration: {int(valid_duration.loc[valid_start].sum())}")
    print(f"Duration source used: {duration_source}")
    if cleaned["duration_hours"].notna().sum() == 0:
        raise ValueError("No positive restoration durations could be created; Gamma model cannot be fit.")

    return cleaned


def normalize_frequency(freq: str) -> str:
    """Normalize common aliases for pandas resampling.

    Pandas has deprecated some uppercase aliases. Normalizing them keeps the
    script compatible with newer pandas versions.
    """

    freq = str(freq).strip()
    if freq.upper() == "H":
        return "h"
    if freq.upper() == "M":
        return "ME"
    return freq


def aggregate_outage_counts(
    df: pd.DataFrame,
    freq: str,
    full_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Aggregate outage starts into a regular time series.

    This creates the observed NHPP increments X_i: the number of outage arrivals
    in each fixed interval.
    """

    freq = normalize_frequency(freq)
    if df.empty:
        if full_index is None:
            return pd.Series(dtype=float, name="outage_count")
        return pd.Series(0, index=full_index, name="outage_count", dtype=float)

    counts = (
        df.set_index("start_timestamp")
        .sort_index()
        .resample(freq)
        .size()
        .astype(float)
        .rename("outage_count")
    )

    if full_index is not None:
        # Scenario subsets are reindexed to the overall timeline so normal and
        # high-disturbance intensities are directly comparable.
        counts = counts.reindex(full_index, fill_value=0)
    else:
        full_index = pd.date_range(counts.index.min(), counts.index.max(), freq=freq)
        counts = counts.reindex(full_index, fill_value=0)
    counts.name = "outage_count"
    return counts


def infer_interval_hours(index: pd.DatetimeIndex, freq: str) -> float:
    """Infer the length of one aggregation interval in hours.

    Availability and lambda(t) both need time in hours, so the aggregation
    interval is converted from the date index/frequency.
    """

    if len(index) > 1:
        diffs = pd.Series(index).diff().dropna().dt.total_seconds() / 3600
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if len(diffs):
            return float(diffs.median())

    freq = normalize_frequency(freq)
    try:
        return pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).total_seconds() / 3600
    except Exception:
        freq_upper = freq.upper()
        if freq_upper.startswith("W"):
            return 24 * 7
        if freq_upper.startswith("ME") or freq_upper.startswith("MS"):
            return 24 * 30.4375
        if freq_upper.startswith("D"):
            return 24
        if freq_upper.startswith("H") or freq == "h":
            return 1
        raise ValueError(f"Could not infer interval length for frequency '{freq}'.")


def floor_timestamps_to_frequency(series: pd.Series, freq: str) -> pd.Series:
    """Floor timestamps to fixed frequencies and approximate non-fixed periods.

    This is only needed by the fallback scenario classifier, where events are
    assigned to high-count periods.
    """

    freq = normalize_frequency(freq)
    try:
        return series.dt.floor(freq)
    except ValueError:
        return series.dt.to_period(freq).dt.to_timestamp()


def estimate_intensity(counts: pd.Series, interval_hours: float) -> pd.DataFrame:
    """Estimate empirical and smoothed NHPP intensity.

    empirical_lambda_per_hour is X_i / Delta_t. smoothed_expected_count is a
    rolling estimate of lambda(t_i) * Delta_t for Poisson simulation.
    """

    # Rolling smoothing reduces random day-to-day count spikes while preserving
    # the observed time-varying outage pattern.
    smoothed_counts = counts.rolling(
        window=ROLLING_WINDOW_INTERVALS,
        center=True,
        min_periods=1,
    ).mean()

    intensity = pd.DataFrame(
        {
            "timestamp": counts.index,
            "outage_count": counts.values,
            "empirical_lambda_per_hour": counts.values / interval_hours,
            "smoothed_expected_count": smoothed_counts.values,
            "smoothed_lambda_per_hour": smoothed_counts.values / interval_hours,
        }
    )
    return intensity


def simulate_nhpp_counts(expected_counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate NHPP interval counts using Poisson(lambda(t_i) * Delta_t).

    The input is expected counts per interval, not restoration duration. This is
    the central separation between the NHPP arrival model and the Gamma recovery
    model.
    """

    expected_counts = np.asarray(expected_counts, dtype=float)
    expected_counts = np.clip(expected_counts, 0, None)
    return rng.poisson(expected_counts)


def fit_gamma_distribution(durations_hours: pd.Series, label: str) -> GammaFit:
    """Fit Gamma(k, theta) restoration-time model and evaluate it.

    Durations must be positive and in hours. The function computes method of
    moments parameters first, then uses scipy's MLE fit with loc fixed at zero
    when possible.
    """

    durations = pd.to_numeric(durations_hours, errors="coerce")
    durations = durations[np.isfinite(durations) & (durations > 0)].to_numpy(dtype=float)

    if len(durations) < 2:
        raise ValueError(f"Need at least two positive durations to fit Gamma model for {label}.")

    empirical_mean = float(np.mean(durations))
    empirical_variance = float(np.var(durations, ddof=1))
    if empirical_variance <= 0:
        # Avoid division by zero if a tiny dataset has effectively constant
        # duration. This keeps the method-of-moments fallback well-defined.
        empirical_variance = max(empirical_mean * 1e-6, 1e-9)

    # Method of moments for Gamma(k, theta):
    # k = mean^2 / variance, theta = variance / mean.
    mom_k = float(empirical_mean**2 / empirical_variance)
    mom_theta = float(empirical_variance / empirical_mean)

    mle_k: float | None = None
    mle_theta: float | None = None
    method = "method_of_moments"
    k = mom_k
    theta = mom_theta
    loc = 0.0

    try:
        # Fixing loc=0 enforces the model definition T_r > 0.
        mle_shape, mle_loc, mle_scale = stats.gamma.fit(durations, floc=0)
        if np.isfinite(mle_shape) and np.isfinite(mle_scale) and mle_shape > 0 and mle_scale > 0:
            mle_k = float(mle_shape)
            mle_theta = float(mle_scale)
            k = mle_k
            theta = mle_theta
            loc = float(mle_loc)
            method = "scipy_mle_floc_0"
    except Exception as exc:
        warnings.warn(f"Gamma MLE fit failed for {label}: {exc}. Using method of moments.")

    # Log-likelihood, AIC, BIC, and KS test summarize Gamma model fit quality.
    logpdf = stats.gamma.logpdf(durations, a=k, loc=loc, scale=theta)
    finite_logpdf = logpdf[np.isfinite(logpdf)]
    log_likelihood = float(np.sum(finite_logpdf)) if len(finite_logpdf) else float("nan")
    n_params = 2
    aic = float(2 * n_params - 2 * log_likelihood) if np.isfinite(log_likelihood) else float("nan")
    bic = float(n_params * np.log(len(durations)) - 2 * log_likelihood) if np.isfinite(log_likelihood) else float("nan")

    try:
        ks_statistic, ks_pvalue = stats.kstest(durations, "gamma", args=(k, loc, theta))
    except Exception as exc:
        warnings.warn(f"KS test failed for {label}: {exc}")
        ks_statistic, ks_pvalue = np.nan, np.nan

    return GammaFit(
        method=method,
        k=float(k),
        theta=float(theta),
        loc=float(loc),
        mom_k=float(mom_k),
        mom_theta=float(mom_theta),
        mle_k=mle_k,
        mle_theta=mle_theta,
        n=int(len(durations)),
        empirical_mean=empirical_mean,
        empirical_variance=empirical_variance,
        fitted_mean=float(k * theta),
        fitted_variance=float(k * theta**2),
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        ks_statistic=float(ks_statistic),
        ks_pvalue=float(ks_pvalue),
    )


def evaluate_nhpp(
    empirical_counts: pd.Series,
    expected_counts: np.ndarray,
    simulated_counts: np.ndarray,
) -> dict[str, float]:
    """Calculate error metrics for empirical vs simulated NHPP counts.

    MAE/RMSE compare interval counts, correlation checks time-pattern agreement,
    and the Poisson log-likelihood evaluates empirical counts under the smoothed
    intensity estimate.
    """

    empirical = empirical_counts.to_numpy(dtype=float)
    simulated = np.asarray(simulated_counts, dtype=float)
    expected = np.asarray(expected_counts, dtype=float)
    expected = np.clip(expected, 1e-12, None)

    # Compare one simulated NHPP path with the empirical count series.
    residual = empirical - simulated
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    if len(empirical) > 1 and np.std(empirical) > 0 and np.std(simulated) > 0:
        correlation = float(np.corrcoef(empirical, simulated)[0, 1])
    else:
        correlation = float("nan")

    poisson_log_likelihood = float(np.sum(stats.poisson.logpmf(empirical, mu=expected)))
    return {
        "mae_empirical_vs_simulated_counts": mae,
        "rmse_empirical_vs_simulated_counts": rmse,
        "correlation_empirical_vs_simulated_counts": correlation,
        "poisson_log_likelihood_empirical_given_smoothed_lambda": poisson_log_likelihood,
    }


def classify_scenarios(cleaned: pd.DataFrame, counts: pd.Series, freq: str) -> tuple[pd.DataFrame, str]:
    """Create normal and high-disturbance flags using cause text or fallback quantiles.

    Cause/event keywords are preferred because they directly represent severe
    disturbances. If the dataset has no usable cause text, high-disturbance is
    inferred from unusually busy periods or unusually long restorations.
    """

    classified = cleaned.copy()
    keyword_pattern = "|".join(re.escape(keyword) for keyword in HIGH_DISTURBANCE_KEYWORDS)

    if "cause_text" in classified.columns and classified["cause_text"].str.strip().ne("").any():
        # Text-based classification is transparent and aligns with the prompt's
        # severe-weather/natural-disaster definition.
        classified["scenario"] = np.where(
            classified["cause_text"].str.contains(keyword_pattern, regex=True, na=False),
            "High-disturbance",
            "Normal",
        )
        method = "cause/event keywords"
    else:
        # Fallback: high-disturbance records are those in the busiest periods
        # or the longest-duration tail when no cause labels exist.
        period_counts = counts.copy()
        count_threshold = float(period_counts.quantile(0.90)) if len(period_counts) else 0
        high_periods = set(period_counts[period_counts >= count_threshold].index)

        period_start = floor_timestamps_to_frequency(classified["start_timestamp"], freq)
        duration_threshold = classified["duration_hours"].quantile(0.90)
        classified["scenario"] = np.where(
            period_start.isin(high_periods) | (classified["duration_hours"] >= duration_threshold),
            "High-disturbance",
            "Normal",
        )
        method = "top outage-count periods or top duration quantile"

    if classified["scenario"].nunique() < 2:
        # Ensure the scenario comparison table is still meaningful even when all
        # records match or miss the keywords.
        duration_threshold = classified["duration_hours"].quantile(0.75)
        classified["scenario"] = np.where(
            classified["duration_hours"] >= duration_threshold,
            "High-disturbance",
            "Normal",
        )
        method += "; duration quantile fallback to ensure both scenarios"

    print_section("Scenario Classification")
    print(f"Method: {method}")
    print(classified["scenario"].value_counts(dropna=False).to_string())
    return classified, method


def calculate_reliability_indicators(
    df: pd.DataFrame,
    total_time_hours: float,
    expected_total_outages: float,
    gamma_fit: GammaFit,
    label: str,
    customer_total_basis: float | None,
) -> dict[str, Any]:
    """Calculate reliability indicators with consistent time units.

    Total downtime and observation time are both measured in hours. This is
    important for the availability approximation A = 1 - D(T) / T.
    """

    durations = df["duration_hours"].dropna()
    durations = durations[(durations > 0) & np.isfinite(durations)]
    total_outages = int(len(df))
    total_downtime = float(durations.sum()) if len(durations) else 0.0
    mean_restoration = float(durations.mean()) if len(durations) else float("nan")
    median_restoration = float(durations.median()) if len(durations) else float("nan")
    restoration_variance = float(durations.var(ddof=1)) if len(durations) > 1 else float("nan")
    expected_downtime = float(expected_total_outages * gamma_fit.k * gamma_fit.theta)
    availability_raw = float(1 - total_downtime / total_time_hours) if total_time_hours > 0 else float("nan")
    availability_clipped = float(np.clip(availability_raw, 0, 1)) if np.isfinite(availability_raw) else float("nan")

    if customer_total_basis and customer_total_basis > 0 and df["customers_affected"].notna().any():
        # The dataset contains affected customers per event but not the true
        # total customer population, so max observed affected customers is only
        # a proxy denominator.
        customer_df = df.loc[df["customers_affected"].notna()].copy()
        saidi = float((customer_df["customers_affected"] * customer_df["duration_hours"].fillna(0)).sum() / customer_total_basis)
        saidi_minutes = saidi * 60
        saifi = float(customer_df["customers_affected"].sum() / customer_total_basis)
        customer_note = (
            "SAIDI/SAIFI use maximum observed affected customers as a proxy for total customers served; "
            "interpret as an approximation because the dataset does not provide a true service-population denominator."
        )
    else:
        saidi = float("nan")
        saidi_minutes = float("nan")
        saifi = float("nan")
        customer_note = "SAIDI/SAIFI skipped because no usable customers-affected column was available."

    limitation_note = ""
    if np.isfinite(availability_raw) and availability_raw < 0:
        # Overlapping or regional outage records can make summed event downtime
        # exceed the observation horizon; clipped availability is reported too.
        limitation_note = (
            "Raw total downtime exceeds the observation horizon. This can happen when outage records overlap, "
            "cover multiple customers/regions, or represent aggregated major events; clipped availability is also reported."
        )

    return {
        "scenario": label,
        "total_outages": total_outages,
        "expected_total_outages_m_T": float(expected_total_outages),
        "observation_time_hours": float(total_time_hours),
        "average_outage_rate_per_day": float(total_outages / (total_time_hours / 24)) if total_time_hours > 0 else float("nan"),
        "mean_restoration_time_hours": mean_restoration,
        "median_restoration_time_hours": median_restoration,
        "restoration_time_variance_hours2": restoration_variance,
        "total_downtime_hours": total_downtime,
        "expected_downtime_hours": expected_downtime,
        "availability_raw": availability_raw,
        "availability_clipped_0_1": availability_clipped,
        "saidi_hours_per_customer": saidi,
        "saidi_minutes_per_customer": saidi_minutes,
        "saifi_interruptions_per_customer": saifi,
        "customer_metric_note": customer_note,
        "availability_limitation_note": limitation_note,
    }


def monte_carlo_downtime(
    expected_counts: np.ndarray,
    gamma_fit: GammaFit,
    total_time_hours: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run Monte Carlo simulations of NHPP arrivals and Gamma recovery durations.

    Each repetition simulates outage counts from the NHPP intensity and then
    samples that many restoration durations from the fitted Gamma distribution.
    """

    rows: list[dict[str, float]] = []
    expected_counts = np.clip(np.asarray(expected_counts, dtype=float), 0, None)

    for simulation_id in range(1, n_simulations + 1):
        # First simulate arrivals N(T), then simulate the recovery time for each
        # outage. This keeps arrival and duration models separate.
        simulated_counts = simulate_nhpp_counts(expected_counts, rng)
        n_outages = int(simulated_counts.sum())
        if n_outages > 0:
            restoration_times = rng.gamma(shape=gamma_fit.k, scale=gamma_fit.theta, size=n_outages)
            total_downtime = float(restoration_times.sum())
        else:
            total_downtime = 0.0
        availability_raw = float(1 - total_downtime / total_time_hours) if total_time_hours > 0 else float("nan")
        rows.append(
            {
                "simulation_id": simulation_id,
                "simulated_outages": n_outages,
                "total_downtime_hours": total_downtime,
                "availability_raw": availability_raw,
                "availability_clipped_0_1": float(np.clip(availability_raw, 0, 1))
                if np.isfinite(availability_raw)
                else float("nan"),
            }
        )

    results = pd.DataFrame(rows)
    downtime = results["total_downtime_hours"]
    availability = results["availability_clipped_0_1"]
    # Percentiles give an uncertainty interval for downtime and availability.
    summary = {
        "downtime_mean_hours": float(downtime.mean()),
        "downtime_median_hours": float(downtime.median()),
        "downtime_std_hours": float(downtime.std(ddof=1)),
        "downtime_p05_hours": float(downtime.quantile(0.05)),
        "downtime_p95_hours": float(downtime.quantile(0.95)),
        "availability_mean_clipped": float(availability.mean()),
        "availability_p05_clipped": float(availability.quantile(0.05)),
        "availability_p95_clipped": float(availability.quantile(0.95)),
    }
    return results, summary


def save_line_plot(
    x: pd.Series | pd.DatetimeIndex | np.ndarray,
    y: pd.Series | np.ndarray,
    title: str,
    y_label: str,
    path: Path,
    color: str = "#1f77b4",
) -> None:
    """Save a single-series line plot."""

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, color=color, linewidth=1.3)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_empirical_vs_smoothed_lambda(intensity: pd.DataFrame, path: Path) -> None:
    """Plot empirical and smoothed NHPP intensity.

    This plot shows the raw X_i / Delta_t estimate and the smoothed lambda(t)
    used for NHPP simulation.
    """

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        intensity["timestamp"],
        intensity["empirical_lambda_per_hour"],
        label="Empirical lambda(t)",
        color="#4c78a8",
        alpha=0.55,
        linewidth=1,
    )
    ax.plot(
        intensity["timestamp"],
        intensity["smoothed_lambda_per_hour"],
        label="Smoothed lambda(t)",
        color="#f58518",
        linewidth=1.8,
    )
    ax.set_title("Empirical vs Smoothed NHPP Intensity")
    ax.set_xlabel("Time")
    ax.set_ylabel("Outages per hour")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cumulative_outages(
    timestamps: pd.DatetimeIndex,
    empirical_counts: pd.Series,
    simulated_counts: np.ndarray,
    path: Path,
) -> None:
    """Plot empirical and simulated cumulative outage counts."""

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, empirical_counts.cumsum(), label="Empirical cumulative N(t)", color="#2f6f4e")
    ax.plot(timestamps, np.cumsum(simulated_counts), label="Simulated cumulative N(t)", color="#c44e52", linestyle="--")
    ax.set_title("Cumulative Outages: Empirical vs Simulated NHPP")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative outage events")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gamma_fit(durations_hours: pd.Series, gamma_fit: GammaFit, path: Path) -> None:
    """Plot empirical duration histogram with fitted Gamma PDF."""

    durations = durations_hours.dropna()
    durations = durations[(durations > 0) & np.isfinite(durations)]
    # Limit the visual range to the 99th percentile so extreme long outages do
    # not compress the main shape of the histogram.
    upper = float(durations.quantile(0.99))
    plot_values = durations[durations <= upper]
    x = np.linspace(max(plot_values.min(), 1e-9), plot_values.max(), 500)
    pdf = stats.gamma.pdf(x, a=gamma_fit.k, loc=gamma_fit.loc, scale=gamma_fit.theta)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(plot_values, bins=40, density=True, alpha=0.65, color="#72b7b2", edgecolor="#ffffff")
    ax.plot(x, pdf, color="#d62728", linewidth=2, label="Fitted Gamma PDF")
    ax.set_title("Recovery Time Histogram with Gamma Fit")
    ax.set_xlabel("Restoration time (hours)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gamma_qq(durations_hours: pd.Series, gamma_fit: GammaFit, path: Path) -> None:
    """Save Q-Q plot for empirical durations against fitted Gamma quantiles."""

    clean_durations = durations_hours.dropna()
    clean_durations = clean_durations[(clean_durations > 0) & np.isfinite(clean_durations)]
    durations = np.sort(clean_durations.to_numpy(dtype=float))
    n = len(durations)
    # Plotting positions convert sorted observations into cumulative
    # probabilities, then Gamma inverse CDF gives the theoretical quantiles.
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.gamma.ppf(probabilities, a=gamma_fit.k, loc=gamma_fit.loc, scale=gamma_fit.theta)

    upper = min(np.nanpercentile(durations, 99), np.nanpercentile(theoretical, 99))
    mask = (durations <= upper) & (theoretical <= upper)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theoretical[mask], durations[mask], s=14, alpha=0.65, color="#4c78a8")
    diagonal_max = max(np.nanmax(theoretical[mask]), np.nanmax(durations[mask]))
    ax.plot([0, diagonal_max], [0, diagonal_max], color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_title("Gamma Q-Q Plot for Restoration Time")
    ax.set_xlabel("Theoretical Gamma quantiles (hours)")
    ax.set_ylabel("Empirical quantiles (hours)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scenario_comparison(scenario_df: pd.DataFrame, path: Path) -> None:
    """Save a multi-panel bar plot comparing scenario indicators."""

    metrics = [
        ("total_outages", "Total outages"),
        ("average_outage_rate_per_day", "Outages/day"),
        ("mean_restoration_time_hours", "Mean restoration (h)"),
        ("total_downtime_hours", "Total downtime (h)"),
        ("expected_downtime_hours", "Expected downtime (h)"),
        ("availability_clipped_0_1", "Availability (clipped)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.ravel()
    colors = {"Normal": "#4c78a8", "High-disturbance": "#e45756"}

    for ax, (metric, title) in zip(axes, metrics):
        plot_data = scenario_df[["scenario", metric]].copy()
        ax.bar(
            plot_data["scenario"],
            plot_data[metric],
            color=[colors.get(name, "#6b7280") for name in plot_data["scenario"]],
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Normal vs High-Disturbance Scenario Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_monte_carlo_downtime(results: pd.DataFrame, summary: dict[str, float], path: Path) -> None:
    """Plot distribution of simulated total downtime."""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(results["total_downtime_hours"], bins=40, color="#59a14f", alpha=0.7, edgecolor="#ffffff")
    ax.axvline(summary["downtime_mean_hours"], color="#1f2937", linewidth=2, label="Mean")
    ax.axvline(summary["downtime_p05_hours"], color="#d62728", linestyle="--", label="5th/95th percentiles")
    ax.axvline(summary["downtime_p95_hours"], color="#d62728", linestyle="--")
    ax.set_title("Monte Carlo Distribution of Total Downtime")
    ax.set_xlabel("Total downtime (hours)")
    ax.set_ylabel("Simulation count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_scenario_models(
    classified: pd.DataFrame,
    full_counts_index: pd.DatetimeIndex,
    interval_hours: float,
    total_time_hours: float,
    overall_gamma: GammaFit,
    customer_total_basis: float | None,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit scenario-specific arrival and duration models, then summarize them.

    Each scenario gets its own NHPP intensity and Gamma restoration model where
    enough data exists. This allows normal and high-disturbance reliability
    indicators to differ in both arrival frequency and recovery duration.
    """

    scenario_rows: list[dict[str, Any]] = []
    scenario_parameters: dict[str, Any] = {}

    for scenario in ["Normal", "High-disturbance"]:
        subset = classified[classified["scenario"] == scenario].copy()
        # Reuse the full timeline so both scenarios have the same observation
        # horizon and interval count.
        scenario_counts = aggregate_outage_counts(
            subset,
            AGGREGATION_FREQUENCY,
            full_index=full_counts_index,
        )
        scenario_intensity = estimate_intensity(scenario_counts, interval_hours)
        expected_counts = scenario_intensity["smoothed_expected_count"].to_numpy(dtype=float)

        try:
            # Fit recovery duration distribution for this scenario only.
            scenario_gamma = fit_gamma_distribution(subset["duration_hours"], scenario)
            gamma_note = "scenario-specific Gamma fit"
        except ValueError as exc:
            warnings.warn(f"{exc} Using overall Gamma parameters for {scenario}.")
            scenario_gamma = overall_gamma
            gamma_note = "overall Gamma fit reused due to insufficient scenario durations"

        scenario_simulated_counts = simulate_nhpp_counts(expected_counts, rng)
        expected_total_outages = float(np.sum(expected_counts))
        # Reliability summary uses observed durations for actual downtime and
        # fitted Gamma parameters for expected downtime.
        indicators = calculate_reliability_indicators(
            subset,
            total_time_hours,
            expected_total_outages,
            scenario_gamma,
            scenario,
            customer_total_basis,
        )
        indicators["simulated_total_outages_one_path"] = int(np.sum(scenario_simulated_counts))
        indicators["gamma_fit_note"] = gamma_note
        scenario_rows.append(indicators)
        scenario_parameters[scenario] = {
            "gamma": scenario_gamma.__dict__,
            "expected_total_outages_m_T": expected_total_outages,
            "simulated_total_outages_one_path": int(np.sum(scenario_simulated_counts)),
            "gamma_fit_note": gamma_note,
        }

    return pd.DataFrame(scenario_rows), scenario_parameters


def save_outputs(
    cleaned: pd.DataFrame,
    counts: pd.Series,
    intensity: pd.DataFrame,
    reliability_summary: pd.DataFrame,
    scenario_comparison: pd.DataFrame,
    monte_carlo_results: pd.DataFrame,
    model_parameters: dict[str, Any],
) -> None:
    """Save all required CSV and JSON output files."""

    cleaned.to_csv(OUTPUT_DIR / "cleaned_outage_data.csv", index=False)
    # The outage count CSV stores empirical counts and both intensity estimates.
    counts_frame = intensity.copy()
    counts_frame.to_csv(OUTPUT_DIR / "outage_count_timeseries.csv", index=False)
    reliability_summary.to_csv(OUTPUT_DIR / "reliability_summary.csv", index=False)
    scenario_comparison.to_csv(OUTPUT_DIR / "scenario_comparison.csv", index=False)
    monte_carlo_results.to_csv(OUTPUT_DIR / "monte_carlo_results.csv", index=False)

    with (OUTPUT_DIR / "model_parameters.json").open("w", encoding="utf-8") as file:
        json.dump(as_serializable(model_parameters), file, indent=2)


def main() -> None:
    """Run the full outage modelling pipeline.

    This function wires together all helper functions in the same order as the
    modelling workflow: load data, clean it, fit models, evaluate them, simulate
    uncertainty, save plots/tables, and print a concise terminal summary.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    # 1. Locate and inspect the dataset before making any assumptions about
    # filenames or column names.
    dataset_path = download_or_find_dataset()
    raw_df, source_file = inspect_and_load_dataset(dataset_path)
    mapping = infer_columns(raw_df)

    # 2. Convert raw records into standardized fields used by every model.
    cleaned = preprocess_data(raw_df, mapping)

    print_section("NHPP Arrival Model")

    # 3. NHPP arrival model:
    #    X_i = number of outages in interval i.
    #    lambda_hat_i = X_i / Delta_t.
    #    smoothed_expected_count approximates lambda(t_i) * Delta_t.
    counts = aggregate_outage_counts(cleaned, AGGREGATION_FREQUENCY)
    interval_hours = infer_interval_hours(counts.index, AGGREGATION_FREQUENCY)
    total_time_hours = float(len(counts) * interval_hours)
    intensity = estimate_intensity(counts, interval_hours)
    expected_counts = intensity["smoothed_expected_count"].to_numpy(dtype=float)

    # One simulated NHPP path is used for empirical-vs-simulated diagnostics.
    simulated_counts = simulate_nhpp_counts(expected_counts, rng)
    nhpp_metrics = evaluate_nhpp(counts, expected_counts, simulated_counts)
    print(f"Aggregation frequency: {normalize_frequency(AGGREGATION_FREQUENCY)}")
    print(f"Intervals: {len(counts)}")
    print(f"Interval length (hours): {interval_hours:.3f}")
    print(f"Total empirical outages N(T): {int(counts.sum())}")
    print(f"Estimated m(T) from smoothed lambda: {float(np.sum(expected_counts)):.3f}")
    print("NHPP evaluation metrics:")
    print(json.dumps(as_serializable(nhpp_metrics), indent=2))

    print_section("Gamma Restoration-Time Model")

    # 4. Gamma duration model:
    #    Only positive restoration durations are used for fitting T_r.
    duration_model_df = cleaned[cleaned["duration_hours"].notna() & (cleaned["duration_hours"] > 0)].copy()
    gamma_fit = fit_gamma_distribution(duration_model_df["duration_hours"], "Overall")
    print(json.dumps(as_serializable(gamma_fit.__dict__), indent=2))

    # Customer metrics need a denominator. This dataset provides affected
    # customers per outage, so the maximum observed value is used as a proxy.
    customer_total_basis = None
    if cleaned["customers_affected"].notna().any():
        customer_total_basis = float(cleaned["customers_affected"].max())
        if not np.isfinite(customer_total_basis) or customer_total_basis <= 0:
            customer_total_basis = None

    # 5. Reliability indicators for the complete dataset.
    reliability_overall = calculate_reliability_indicators(
        cleaned,
        total_time_hours,
        float(np.sum(expected_counts)),
        gamma_fit,
        "Overall",
        customer_total_basis,
    )
    reliability_summary = pd.DataFrame([reliability_overall])

    print_section("Scenario Modelling")

    # 6. Split events into normal and high-disturbance scenarios, then fit
    # scenario-specific NHPP/Gamma models.
    classified, scenario_method = classify_scenarios(cleaned, counts, AGGREGATION_FREQUENCY)
    scenario_comparison, scenario_parameters = run_scenario_models(
        classified,
        counts.index,
        interval_hours,
        total_time_hours,
        gamma_fit,
        customer_total_basis,
        rng,
    )
    print(scenario_comparison.to_string(index=False))

    print_section("Monte Carlo Simulation")

    # 7. Repeated simulations estimate the distribution of total downtime and
    # availability under the fitted NHPP + Gamma model.
    monte_carlo_results, monte_carlo_summary = monte_carlo_downtime(
        expected_counts,
        gamma_fit,
        total_time_hours,
        N_MONTE_CARLO_SIMULATIONS,
        rng,
    )
    print(json.dumps(as_serializable(monte_carlo_summary), indent=2))

    print_section("Saving Plots")

    # 8. Save all required visual diagnostics into outputs/.
    save_line_plot(
        counts.index,
        counts.values,
        "Empirical Outage Counts Over Time",
        "Outage count",
        OUTPUT_DIR / "outage_count_timeseries.png",
    )
    plot_empirical_vs_smoothed_lambda(intensity, OUTPUT_DIR / "empirical_vs_smoothed_lambda.png")
    plot_cumulative_outages(
        counts.index,
        counts,
        simulated_counts,
        OUTPUT_DIR / "cumulative_outages_empirical_vs_simulated.png",
    )
    plot_gamma_fit(duration_model_df["duration_hours"], gamma_fit, OUTPUT_DIR / "recovery_time_histogram_gamma_fit.png")
    plot_gamma_qq(duration_model_df["duration_hours"], gamma_fit, OUTPUT_DIR / "qq_plot_gamma_recovery_time.png")
    plot_scenario_comparison(scenario_comparison, OUTPUT_DIR / "scenario_comparison_barplot.png")
    plot_monte_carlo_downtime(monte_carlo_results, monte_carlo_summary, OUTPUT_DIR / "simulated_downtime_distribution.png")
    print(f"Plots saved to: {OUTPUT_DIR.resolve()}")

    # 9. Capture parameters and diagnostics in a JSON-friendly dictionary for
    # reproducibility and auditing.
    model_parameters = {
        "dataset_slug": DATASET_SLUG,
        "source_file": source_file,
        "column_mapping": mapping.__dict__,
        "aggregation_frequency": normalize_frequency(AGGREGATION_FREQUENCY),
        "rolling_window_intervals": ROLLING_WINDOW_INTERVALS,
        "random_seed": RANDOM_SEED,
        "interval_hours": interval_hours,
        "total_time_hours": total_time_hours,
        "nhpp": {
            "m_T": float(np.sum(expected_counts)),
            "average_empirical_count_per_interval": float(counts.mean()),
            "average_smoothed_lambda_per_hour": float(intensity["smoothed_lambda_per_hour"].mean()),
            "evaluation": nhpp_metrics,
        },
        "gamma": gamma_fit.__dict__,
        "scenario_method": scenario_method,
        "scenarios": scenario_parameters,
        "monte_carlo": monte_carlo_summary,
        "notes": {
            "duration_unit": "hours",
            "availability": "availability_raw = 1 - total_downtime_hours / observation_time_hours; clipped availability is bounded to [0, 1].",
            "saidi_saifi": reliability_overall["customer_metric_note"],
        },
    }

    # 10. Save all required tabular outputs and model metadata.
    save_outputs(
        classified,
        counts,
        intensity,
        reliability_summary,
        scenario_comparison,
        monte_carlo_results,
        model_parameters,
    )

    print_section("Final Summary")

    # 11. Print concise terminal results so the script is useful even before the
    # user opens the output files.
    print(f"Source file: {source_file}")
    print(f"Cleaned rows with valid start timestamp: {len(cleaned)}")
    print(f"Rows used for Gamma duration model: {len(duration_model_df)}")
    print(f"Total outage events N(T): {reliability_overall['total_outages']}")
    print(f"Average outage rate: {reliability_overall['average_outage_rate_per_day']:.4f} outages/day")
    print(f"Mean restoration time: {reliability_overall['mean_restoration_time_hours']:.4f} hours")
    print(f"Expected downtime E[D(T)]: {reliability_overall['expected_downtime_hours']:.4f} hours")
    print(f"Availability raw: {reliability_overall['availability_raw']:.6f}")
    print(f"Availability clipped: {reliability_overall['availability_clipped_0_1']:.6f}")
    if reliability_overall["availability_limitation_note"]:
        print(f"Availability note: {reliability_overall['availability_limitation_note']}")
    print(f"All output files saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
