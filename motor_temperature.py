"""Estimate stator winding temperature from 2 Hz motor measurements."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


TARGET = "stator_winding"
PROFILE = "profile_id"
SENSORS = ["ambient", "coolant", "i_d", "i_q", "u_d", "u_q", "motor_speed"]
HISTORY = ["current_squared", "speed_abs", "coolant"]
WINDOWS = [60, 300]
SAMPLE_SECONDS = 0.5
STRIDE = 10
FILL_LIMIT = 10

INSTANT = SENSORS + ["current_squared", "speed_abs"]
FEATURES = INSTANT + [f"{name}_mean_{s}s" for name in HISTORY for s in WINDOWS]
MODEL_FILE = "motor_temperature_model.joblib"
FEATURE_SETTINGS = {
    "base_columns": SENSORS, "history_columns": HISTORY, "windows_seconds": WINDOWS,
    "sample_seconds": SAMPLE_SECONDS, "stride": STRIDE, "forward_fill_limit": FILL_LIMIT,
}


def read_measurements(path, labelled):
    numeric = SENSORS + ([TARGET] if labelled else [])
    required = [PROFILE, *numeric]
    data = pd.read_csv(path, usecols=lambda col: col in required, dtype={PROFILE: "string"},
                       low_memory=False)
    missing = [col for col in required if col not in data]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError(f"{path.name} has no readings.")
    labels = data[PROFILE].str.strip()
    if labels.isna().any() or labels.eq("").any():
        raise ValueError("Every reading needs a profile_id so its history can be traced.")
    # Plain numeric IDs use the notebook's split order. Keep labels such as "01" distinct.
    unique = labels.drop_duplicates()
    numbers = pd.to_numeric(unique, errors="coerce")
    if numbers.notna().all() and numbers.astype(str).tolist() == unique.tolist():
        labels = pd.to_numeric(labels)
    data[PROFILE] = labels
    values = data[numeric].apply(pd.to_numeric, errors="coerce").astype("float32")
    data[numeric] = values.replace([np.inf, -np.inf], np.nan)
    return data


def hold_out(data, fraction, seed):
    split = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    keep, held = next(split.split(data, groups=data[PROFILE]))
    return data.iloc[keep], data.iloc[held]


def input_medians(data):
    medians = data[SENSORS].median()
    empty = medians.index[medians.isna()].tolist()
    if empty:
        raise ValueError(f"No usable training readings for {', '.join(empty)}.")
    return medians


def make_features(raw, medians):
    """Start fresh at each profile and build the history before taking every tenth row."""
    parts = []
    for _, profile in raw.groupby(PROFILE, sort=False):
        profile = profile.copy()
        profile[SENSORS] = profile[SENSORS].ffill(limit=FILL_LIMIT).fillna(medians)
        currents = profile[["i_d", "i_q"]].astype("float64")
        profile["current_squared"] = currents["i_d"] ** 2 + currents["i_q"] ** 2
        profile["speed_abs"] = profile["motor_speed"].abs()
        for seconds in WINDOWS:
            samples = round(seconds / SAMPLE_SECONDS)
            means = profile[HISTORY].rolling(samples, min_periods=1).mean()
            profile = profile.join(means.add_suffix(f"_mean_{seconds}s"))
        kept = profile.iloc[::STRIDE].copy()
        kept["elapsed_s"] = np.arange(len(kept)) * STRIDE * SAMPLE_SECONDS
        parts.append(kept)
    return pd.concat(parts).sort_index()


def labelled_features(raw, medians, split_name):
    data = make_features(raw, medians).dropna(subset=[TARGET])
    if len(data) < 2:
        raise ValueError(f"The {split_name} split has fewer than two labelled rows after sampling.")
    return data


def model_options(seed):
    linear = make_pipeline(StandardScaler(), LinearRegression())
    forest = RandomForestRegressor(
        n_estimators=100, max_depth=12, min_samples_leaf=10,
        max_features=0.8, n_jobs=-1, random_state=seed,
    )
    options = {
        "Mean baseline": (DummyRegressor(strategy="mean"), FEATURES),
        "Linear regression": (linear, FEATURES),
        "Linear regression (no history)": (clone(linear), INSTANT),
    }
    for alpha in (0.1, 1, 10, 100):
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        options[f"Ridge (alpha={alpha})"] = ridge, FEATURES
    tree = DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=seed)
    options["Decision tree"] = tree, FEATURES
    options["Random forest"] = forest, FEATURES
    options["Random forest (no history)"] = clone(forest), INSTANT
    return options


def scores(actual, predicted):
    return {
        "mae_c": mean_absolute_error(actual, predicted),
        "rmse_c": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": r2_score(actual, predicted),
    }


def compare_models(train, validation, seed):
    options = model_options(seed)
    rows = []
    for name, (model, columns) in options.items():
        print(f"Fitting {name} ...", flush=True)
        model.fit(train[columns], train[TARGET])
        predicted = model.predict(validation[columns])
        rows.append({"model": name, **scores(validation[TARGET], predicted)})
    table = pd.DataFrame(rows).sort_values("rmse_c", kind="stable")
    name = table.iloc[0]["model"]
    model, columns = options[name]
    return table, name, clone(model), columns


def plot_profile(profile, path):
    fig = Figure(figsize=(9, 5), layout="constrained")
    temp, error = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])
    minutes = profile["elapsed_s"] / 60
    profile_id = profile[PROFILE].iloc[0]
    mae = profile["error_c"].abs().mean()
    temp.set_title(f"Test session {profile_id}  |  Highest mean absolute error ({mae:.2f} °C)")
    temp.plot(minutes, profile[TARGET], label="Measured", color="#23566b", linewidth=1.7)
    temp.plot(minutes, profile["predicted_c"], label="Estimated", color="#c15f35", linewidth=1.3)
    temp.set_ylabel("Winding temperature (°C)")
    temp.legend(frameon=False, ncols=2)
    error.axhline(0, color="#68737a", linewidth=0.8)
    error.plot(minutes, profile["error_c"], color="#75466e", linewidth=1)
    error.set(xlabel="Minutes into session", ylabel="Error (°C)")
    for ax in (temp, error):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=150)


def publish_run(stage, output):
    """Move a complete run into place, keeping earlier runs in their own folders."""
    output = output.absolute()
    folder = output
    number = 1
    while True:
        if not folder.exists():
            try:
                return stage.rename(folder)
            except OSError:
                if not folder.exists():
                    raise
        number += 1
        folder = output.with_name(f"{output.name}-{number}")


def train_model(csv, output, seed):
    raw = read_measurements(csv, labelled=True)
    if raw[PROFILE].nunique() < 3:
        raise ValueError("Training needs at least three profiles for train, validation and test.")
    dev, test_raw = hold_out(raw, 0.15, seed)
    train_raw, val_raw = hold_out(dev, 0.15 / 0.85, seed)
    splits = {"train": train_raw, "validation": val_raw, "test": test_raw}
    profiles = {name: sorted(part[PROFILE].unique().tolist()) for name, part in splits.items()}
    print("Sessions   " + "   ".join(f"{name} {len(ids)}" for name, ids in profiles.items()))

    medians = input_medians(train_raw)
    train = labelled_features(train_raw, medians, "training")
    val = labelled_features(val_raw, medians, "validation")
    table, name, model, columns = compare_models(train, val, seed)

    # The model choice is fixed. Refit the medians and estimator on both development splits.
    medians = input_medians(dev)
    fit_data = labelled_features(dev, medians, "combined training and validation")
    model.fit(fit_data[columns], fit_data[TARGET])
    test = labelled_features(test_raw, medians, "test")
    predictions = test[[PROFILE, "elapsed_s", TARGET]].copy()
    predictions["predicted_c"] = model.predict(test[columns])
    predictions["error_c"] = predictions["predicted_c"] - predictions[TARGET]
    test_scores = scores(predictions[TARGET], predictions["predicted_c"])
    sessions = predictions.groupby(PROFILE, sort=False)["error_c"].agg(
        readings="size",
        mae_c=lambda errors: errors.abs().mean(),
        rmse_c=lambda errors: np.sqrt((errors ** 2).mean()),
    ).sort_values("mae_c", ascending=False)
    bundle = {
        "model": model, "model_name": name, "features": columns,
        "input_medians": medians.to_dict(), "sklearn_version": sklearn.__version__,
        **FEATURE_SETTINGS,
    }
    summary = {
        "model": name, "seed": seed, "split_sessions": profiles, "features": columns,
        "validation_rmse_c": float(table.iloc[0]["rmse_c"]),
        "test_rows": len(test), "test_scores": test_scores, "sklearn_version": sklearn.__version__,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "run"
        stage.mkdir()
        table.to_csv(stage / "validation_models.csv", index=False)
        predictions.to_csv(stage / "test_predictions.csv", index=False)
        sessions.to_csv(stage / "test_session_scores.csv")
        worst = predictions[predictions[PROFILE] == sessions.index[0]]
        plot_profile(worst, stage / "temperature_trace.png")
        (stage / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        joblib.dump(bundle, stage / MODEL_FILE)
        output = publish_run(stage, output)

    print("\nValidation scores")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n{name} on {len(test):,} held-out readings")
    print(f"MAE {test_scores['mae_c']:.2f} °C   RMSE {test_scores['rmse_c']:.2f} °C   "
          f"R² {test_scores['r2']:.4f}")
    print(f"Results saved to {output}")


def predict(csv, model_path, output):
    if output.resolve() in {csv.resolve(), model_path.resolve()}:
        raise ValueError("Choose a prediction output path separate from the input CSV and model.")
    raw = read_measurements(csv, labelled=False)
    saved = joblib.load(model_path)
    if not isinstance(saved, dict) or any(saved.get(k) != v for k, v in FEATURE_SETTINGS.items()):
        raise ValueError("The saved model uses a different set of feature settings.")
    ready = make_features(raw, pd.Series(saved["input_medians"]))
    predictions = ready[[PROFILE, "elapsed_s"]].copy()
    predictions["predicted_c"] = saved["model"].predict(ready[saved["features"]])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        path = Path(tmp) / "estimates.csv"
        predictions.to_csv(path, index=False)
        path.replace(output)
    print(f"Saved {len(predictions):,} estimates to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Compare models and score the best on held-out sessions.")
    train.add_argument("csv", type=Path, help="Labelled measurements in acquisition order.")
    train.add_argument("--out", type=Path, default=Path("results"), help="Run folder (default: results).")
    train.add_argument("--seed", type=int, default=42, help="Split and model seed (default: 42).")
    estimate = commands.add_parser("predict", help="Estimate winding temperature for new sessions.")
    estimate.add_argument("csv", type=Path, help="Sensor readings with a profile_id for each row.")
    estimate.add_argument("--model", type=Path, required=True, help="Model saved by a training run.")
    estimate.add_argument("--out", type=Path, default=Path("predictions.csv"), help="Output CSV path.")
    args = parser.parse_args()
    try:
        if args.command == "train":
            train_model(args.csv, args.out, args.seed)
        else:
            predict(args.csv, args.model, args.out)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
