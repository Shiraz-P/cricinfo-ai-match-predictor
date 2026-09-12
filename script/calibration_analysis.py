import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss
)


# -------------------------------------------------
# Load Feature Dataset
# -------------------------------------------------

file_path = r"K:\Python\Cricinfo_AI_Project\data\matches_features.csv"

df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# -------------------------------------------------
# Features
# -------------------------------------------------

features = [
    "toss_winner_is_team1",
    "team1_historical_win_rate",
    "team2_historical_win_rate",
    "team1_recent_win_rate",
    "team2_recent_win_rate",
    "team1_h2h_win_rate",
    "team2_h2h_win_rate",
    "team1_venue_win_rate",
    "team2_venue_win_rate"
]

X = df[features]
y = df["team1_won"]


# -------------------------------------------------
# Chronological Train / Test Split
# -------------------------------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# -------------------------------------------------
# Train Logistic Regression
# -------------------------------------------------

model = LogisticRegression()

model.fit(
    X_train,
    y_train
)


# -------------------------------------------------
# Predictions and Probabilities
# -------------------------------------------------

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# -------------------------------------------------
# Accuracy
# -------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)


# -------------------------------------------------
# Model Brier Score
# Lower is better
# -------------------------------------------------

model_brier = brier_score_loss(
    y_test,
    probabilities
)


# -------------------------------------------------
# Baseline Probability
#
# Baseline predicts the training-set Team 1 win rate
# for every test match.
#
# Important:
# We use TRAINING data only so the baseline does not
# learn information from the test period.
# -------------------------------------------------

baseline_probability = y_train.mean()

baseline_probabilities = np.full(
    len(y_test),
    baseline_probability
)

baseline_brier = brier_score_loss(
    y_test,
    baseline_probabilities
)


# -------------------------------------------------
# Brier Improvement
# -------------------------------------------------

brier_improvement = (
    baseline_brier
    -
    model_brier
)

brier_improvement_percentage = (
    brier_improvement
    /
    baseline_brier
    *
    100
)


# -------------------------------------------------
# Main Summary
# -------------------------------------------------

print("\n================================")
print("PROBABILITY CALIBRATION ANALYSIS")
print("================================")

print(
    "Test matches:",
    len(X_test)
)

print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)

print(
    "Model Brier Score:",
    round(
        model_brier,
        4
    )
)

print(
    "Baseline probability:",
    round(
        baseline_probability * 100,
        2
    ),
    "%"
)

print(
    "Baseline Brier Score:",
    round(
        baseline_brier,
        4
    )
)

print(
    "Brier Score improvement:",
    round(
        brier_improvement,
        4
    )
)

print(
    "Brier improvement percentage:",
    round(
        brier_improvement_percentage,
        2
    ),
    "%"
)


# -------------------------------------------------
# Build Calibration Dataset
# -------------------------------------------------

calibration_df = pd.DataFrame({
    "actual": y_test.values,
    "predicted_probability": probabilities
})


# -------------------------------------------------
# Create Probability Bins
#
# 0.0 - 0.1
# 0.1 - 0.2
# ...
# 0.9 - 1.0
# -------------------------------------------------

bin_edges = np.linspace(
    0,
    1,
    11
)

calibration_df["probability_bin"] = pd.cut(
    calibration_df["predicted_probability"],
    bins=bin_edges,
    include_lowest=True
)


# -------------------------------------------------
# Analyze Each Bin
# -------------------------------------------------

bin_results = (
    calibration_df
    .groupby(
        "probability_bin",
        observed=True
    )
    .agg(
        matches=(
            "actual",
            "size"
        ),
        average_predicted_probability=(
            "predicted_probability",
            "mean"
        ),
        actual_win_rate=(
            "actual",
            "mean"
        )
    )
    .reset_index()
)


# -------------------------------------------------
# Calibration Difference
#
# Positive:
# actual win rate > predicted probability
# Model underestimated Team 1 probability.
#
# Negative:
# actual win rate < predicted probability
# Model overestimated Team 1 probability.
# -------------------------------------------------

bin_results["difference"] = (
    bin_results["actual_win_rate"]
    -
    bin_results["average_predicted_probability"]
)


# -------------------------------------------------
# Absolute Calibration Error
# -------------------------------------------------

bin_results["absolute_difference"] = (
    bin_results["difference"].abs()
)


# -------------------------------------------------
# Weighted Calibration Error
#
# Gives larger bins more influence than tiny bins.
# -------------------------------------------------

bin_results["weight"] = (
    bin_results["matches"]
    /
    len(calibration_df)
)

weighted_calibration_error = (
    bin_results["absolute_difference"]
    *
    bin_results["weight"]
).sum()


# -------------------------------------------------
# Prepare Display Table
# -------------------------------------------------

display_df = bin_results.copy()

display_df[
    "average_predicted_probability"
] = (
    display_df[
        "average_predicted_probability"
    ]
    * 100
).round(2)

display_df[
    "actual_win_rate"
] = (
    display_df[
        "actual_win_rate"
    ]
    * 100
).round(2)

display_df[
    "difference"
] = (
    display_df[
        "difference"
    ]
    * 100
).round(2)

display_df[
    "absolute_difference"
] = (
    display_df[
        "absolute_difference"
    ]
    * 100
).round(2)


# -------------------------------------------------
# Display Calibration Bins
# -------------------------------------------------

print(
    "\n--- CALIBRATION BINS ---"
)

print(
    display_df[
        [
            "probability_bin",
            "matches",
            "average_predicted_probability",
            "actual_win_rate",
            "difference",
            "absolute_difference"
        ]
    ].to_string(
        index=False
    )
)


# -------------------------------------------------
# Weighted Calibration Error
# -------------------------------------------------

print(
    "\n--- CALIBRATION ERROR ---"
)

print(
    "Weighted Absolute Calibration Error:",
    round(
        weighted_calibration_error * 100,
        2
    ),
    "percentage points"
)


# -------------------------------------------------
# Largest Calibration Difference
# -------------------------------------------------

largest_error_row = bin_results.loc[
    bin_results[
        "absolute_difference"
    ].idxmax()
]

print(
    "\n--- LARGEST BIN CALIBRATION ERROR ---"
)

print(
    "Probability Bin:",
    largest_error_row[
        "probability_bin"
    ]
)

print(
    "Matches:",
    largest_error_row[
        "matches"
    ]
)

print(
    "Average Predicted Probability:",
    round(
        largest_error_row[
            "average_predicted_probability"
        ] * 100,
        2
    ),
    "%"
)

print(
    "Actual Win Rate:",
    round(
        largest_error_row[
            "actual_win_rate"
        ] * 100,
        2
    ),
    "%"
)

print(
    "Absolute Difference:",
    round(
        largest_error_row[
            "absolute_difference"
        ] * 100,
        2
    ),
    "percentage points"
)


# -------------------------------------------------
# Interpretation
# -------------------------------------------------

print(
    "\n--- INTERPRETATION ---"
)

if model_brier < baseline_brier:

    print(
        "The model has a better Brier Score "
        "than the training-rate baseline."
    )

elif model_brier > baseline_brier:

    print(
        "The model has a worse Brier Score "
        "than the training-rate baseline."
    )

else:

    print(
        "The model and baseline have the same "
        "Brier Score."
    )


print(
    "\nLower Brier Score is better."
)

print(
    "A Brier Score of 0 represents perfect "
    "probability predictions."
)

print(
    "\nCalibration difference = "
    "Actual Win Rate - Predicted Probability."
)

print(
    "Positive difference means the model "
    "underestimated Team 1's probability."
)

print(
    "Negative difference means the model "
    "overestimated Team 1's probability."
)

print(
    "\nBin match counts are important."
)

print(
    "A large calibration difference based on "
    "very few matches should be interpreted "
    "more cautiously."
)