import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Optional, Union, List

# ------------------------------------------------------
# Column mapping
# ------------------------------------------------------
COLUMN_MAP = {
    "KTH_A0043032_AM101_VAD_GM91_MV": "rh[-]",                          # Relative humidity
    "KTH_A0043032_AM101_VAD_GRADI_MV": "rad[W/m^2]",                    # Global radiation (solar)
    "KTH_A0043032_AM101_VAD_GS91_MV": "wind_speed[m/s]",                # Wind speed
    "KTH_A0043032_AM101_VAD_GT91_MV": "air_temp[C]",                    # Outdoor air temperature
    "KTH_A0043032_AM101_VAD_VINDD_MV": "wind_dir[deg]",                 # Wind direction
    "KTH_A0043032_KP101_120_MO401_EF": "cooling_power[kW]",             # Space cooling power (kW)
    "KTH_A0043032_KP101_120_MO401_FM": "energy_carrier_cooling[m^3/h]", # Energy carrier cooling (m^3/h)
    "KTH_A0043032_VP101_120_MO401_FM": "energy_carrier_heating[m^3/h]", # Energy carrier heating (m^3/h)
    "KTH_A0043032_VP101_120_MO401_EF": "heat_power[kW]",                # Space heating power (kW)
}

# Columns that must be non-negative
POSITIVE_COLUMNS = [
    "energy_carrier_cooling[m^3/h]",
    "energy_carrier_heating[m^3/h]",
    "cooling_power[kW]",
    "heat_power[kW]",
]

# columns to clamp the outliers for physical validity
PERCENTILE_CLAMP_VALUES = {
    "rad[W/m^2]": (0, 95),
    "wind_speed[m/s]": (0, 99),
    "cooling_power[kW]": (0, 99),
    "heat_power[kW]": (0, 99),
    "energy_carrier_cooling[m^3/h]": (0, 99),
    "energy_carrier_heating[m^3/h]": (0, 99),
}

# ------------------------------------------------------
# Utility functions
# ------------------------------------------------------
def _find_datetime_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "date" in c.lower()]
    if not candidates:
        raise ValueError("No datetime column found (expected a 'Date' column).")
    return candidates[0]


def load_raw(csv_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_and_concat_folder(folder_path: Union[str, Path], pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load and concatenate all CSVs from a folder into one DataFrame.
    Assumes all have the same schema.
    """
    folder_path = Path(folder_path)
    csv_files: List[Path] = sorted(folder_path.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")

    dfs = []
    for f in csv_files:
        print(f"[INFO] Loading {f.name}")
        df = load_raw(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Concatenated {len(csv_files)} CSVs → {df_all.shape[0]} rows")
    return df_all


# ------------------------------------------------------
# Preprocessing
# ------------------------------------------------------
def normalize_wind_dir(df: pd.DataFrame, col: str = "wind_dir") -> pd.DataFrame:
    """
    Normalize wind direction to [0, 360) degrees.
    """
    df[col] = df[col] % 360
    return df


def handle_physical_violations(
    df: pd.DataFrame,
    column: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    strategy: str = "set_nan_and_flag",
    flag_suffix: str = "_phys_violation",
) -> pd.DataFrame:
    """
    Handle physically impossible values in a given column using common strategies.

    Args:
        df: input DataFrame (will NOT be modified in place; a copy is returned).
        column: name of the column to clean (e.g. "heat_power").
        min_val: minimum physically valid value (inclusive), or None for no lower bound.
        max_val: maximum physically valid value (inclusive), or None for no upper bound.
        strategy:
            - "clip":
                values < min_val -> min_val
                values > max_val -> max_val
            - "set_nan":
                out-of-range values -> NaN
            - "drop":
                drop rows where values are out-of-range
            - "flag":
                keep original values; add boolean flag column (True if invalid)
            - "set_nan_and_flag":
                out-of-range -> NaN AND add flag column
        flag_suffix:
            suffix for the created flag column, e.g. "heat_power_phys_violation".

    Returns:
        A new DataFrame with cleaned column (and possibly an added flag column).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    clean_df = df.copy()

    vals = clean_df[column].astype(float)

    invalid_mask = pd.Series(False, index=clean_df.index)
    if min_val is not None:
        invalid_mask |= vals < min_val
    if max_val is not None:
        invalid_mask |= vals > max_val

    if not invalid_mask.any():
        # nothing to do
        return clean_df

    flag_col = f"{column}{flag_suffix}"

    if strategy == "clip":
        if min_val is not None:
            vals = vals.mask(vals < min_val, min_val)
        if max_val is not None:
            vals = vals.mask(vals > max_val, max_val)
        clean_df[column] = vals

    elif strategy == "set_nan":
        vals = vals.mask(invalid_mask, np.nan)
        clean_df[column] = vals

    elif strategy == "drop":
        clean_df = clean_df.loc[~invalid_mask].copy()

    elif strategy == "flag":
        clean_df[flag_col] = invalid_mask

    elif strategy == "set_nan_and_flag":
        vals = vals.mask(invalid_mask, np.nan)
        clean_df[column] = vals
        clean_df[flag_col] = invalid_mask

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from ['clip', 'set_nan', 'drop', 'flag', 'set_nan_and_flag']."
        )

    return clean_df


def preprocess_day_ahead_heating(
    input_path: Union[str, Path],
    output_csv_path: Optional[Union[str, Path]] = None,
    daily_agg: Literal["mean", "sum"] = "mean",
    min_fraction_per_day: float = 0.5,
) -> pd.DataFrame:
    """
    Preprocess raw or folder of CSVs into a day-ahead supervised dataset
    ready for MLP training.

    If `input_path` is a folder, all CSVs inside will be concatenated.
    """
    input_path = Path(input_path)

    # 1. Load
    if input_path.is_dir():
        df = load_and_concat_folder(input_path)
    else:
        df = load_raw(input_path)

    # 2. Parse datetime
    dt_col = _find_datetime_col(df)
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])
    df = df.sort_values(dt_col)

    # 3. Rename columns (keep only necessary)
    rename_map = {raw: clean for raw, clean in COLUMN_MAP.items() if raw in df.columns}
    missing = [raw for raw in COLUMN_MAP.keys() if raw not in df.columns]
    if missing:
        print(f"[WARN] Missing expected columns (will be skipped): {missing}")

    df = df[[dt_col] + list(rename_map.keys())].rename(columns=rename_map)

    # 4. Convert numeric
    for c in df.columns:
        if c != dt_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.set_index(dt_col)

    # 5. Aggregate daily
    agg_funcs = {c: daily_agg for c in ["rh", "rad", "wind_speed", "wind_dir", "air_temp", "heat_power"] if c in df.columns}
    daily = df.resample("D").agg(agg_funcs)

    # 6. Drop very incomplete days
    counts = df.resample("D").count()
    max_count = counts.max().max()
    min_required = max_count * min_fraction_per_day if max_count > 0 else 1
    valid_days = counts.max(axis=1) >= min_required
    daily = daily[valid_days]

    # 7. Fill small gaps
    daily = daily.interpolate(limit=2, limit_direction="both")
    daily = daily.dropna(subset=["heat_power"])

    # 8. Build target (next-day heating)
    daily["target_heating_next_day"] = daily["heat_power"].shift(-1)

    final_df = daily[["rh", "rad", "wind_speed", "wind_dir", "air_temp", "target_heating_next_day"]].copy()
    final_df = final_df.dropna(subset=["target_heating_next_day"]).reset_index()
    final_df = final_df.rename(columns={"index": "date"})

    # 9. Save
    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_csv_path, index=False)
        print(f"[INFO] Saved preprocessed dataset to: {output_csv_path}")

    return final_df
