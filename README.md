# Motor winding temperature estimation

A motor can draw the same current twice and have a hotter winding the second time. I tested how much the last five minutes of operation help estimate the present stator winding temperature of a permanent-magnet synchronous motor.

## Results from the original notebook

The full run used 1,330,816 readings from 69 sessions. With seed 42, 47 sessions went into training, 11 into validation and 11 into testing. Linear regression narrowly beat Ridge (`alpha=0.1`) on validation RMSE.

| Model | Test readings | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Linear regression | 26,726 | 4.07 °C | 5.84 °C | 0.9644 |

For the random forest, adding six trailing averages reduced validation RMSE from **14.16 °C to 6.75 °C**. The test scores above describe new sessions on the same motor.

## Run it

Download and extract `measures_v2.csv` beside the script. Paderborn University’s LEA department recorded the measurements at 2 Hz. [Electric Motor Temperature dataset](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature) · [Data licence (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)

From a Python 3.12 environment, run

```bash
python -m pip install numpy==2.3.5 pandas==2.2.3 scikit-learn==1.8.0 matplotlib==3.10.8 joblib==1.5.3
python motor_temperature.py train measures_v2.csv
```

In Colab, upload the script and CSV together, then run `!python motor_temperature.py train measures_v2.csv` in a cell.

## What goes into the estimate

| CSV columns | Purpose |
| --- | --- |
| `i_d`, `i_q`, `u_d`, `u_q` | Direct- and quadrature-axis currents and voltages |
| `motor_speed`, `ambient`, `coolant` | Shaft speed, ambient temperature and coolant temperature |
| `profile_id` | Identifies the session and marks where its history starts |
| `stator_winding` | Measured winding temperature in °C, used during training and evaluation |

The script adds squared current magnitude (`i_d² + i_q²`) as a heating proxy, plus absolute speed. Trailing averages of squared current, absolute speed and coolant temperature over 60 and 300 seconds bring the total to 15 features. Each session starts with its own history.

Each sensor carries forward its last reading for up to ten samples. Any remaining gaps use training medians. History is calculated at 2 Hz before keeping one reading every five seconds. Rows with a missing target are skipped for fitting and scoring.

Validation RMSE selects among a mean baseline, linear regression, Ridge, a decision tree and a random forest. Both linear regression and the forest run with and without the history features. The winning model and its preprocessing are refitted on training plus validation sessions before the final test.

## Inspect a run

The first run writes to `results/`. Later runs get numbered folders. Use `--out` to name a run or `--seed` to try a different split.

| File | What to look for |
| --- | --- |
| `temperature_trace.png` | Measured and estimated temperatures in the test session with the highest MAE. Positive errors in the lower panel mean the estimate ran hot. |
| `validation_models.csv` | Validation scores for every model, including both history comparisons |
| `test_session_scores.csv`, `test_predictions.csv` | Error by session and the individual readings behind each score |
| `run_summary.json` | Selected model, feature names, split IDs and overall test metrics |
| `motor_temperature_model.joblib` | Fitted model and preprocessing settings for reuse |

## Estimate a new session

```bash
python motor_temperature.py predict new_session.csv --model results/motor_temperature_model.joblib --out estimates.csv
```

Supply the seven sensor columns and `profile_id`. Keep the 0.5-second rows in acquisition order within each session, including rows with a missing sensor value. The script infers time from row position and writes `profile_id`, `elapsed_s` and `predicted_c` at five-second intervals, starting at zero.
