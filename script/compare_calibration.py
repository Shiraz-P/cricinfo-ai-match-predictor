import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss
)


# -------------------------------------------------
# Load Dataset
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
# Final chronological test split
#
# First 80% = model development
# Last 20% = untouched final test
# -------------------------------------------------

final_split_index = int(len(df) * 0.8)

X_development = X.iloc[:final_split_index]
y_development = y.iloc[:final_split_index]

X_test = X.iloc[final_split_index:]
y_test = y.iloc[final_split_index:]


# -------------------------------------------------
# Split development data into:
# 80% training
# 20% calibration
# -------------------------------------------------

calibration_split_index = int(
    len(X_development) * 0.8
)

X_train = X_development.iloc[
    :calibration_split_index
]

y_train = y_development.iloc[
    :calibration_split_index
]

X_calibration = X_development.iloc[
    calibration_split_index:
]

y_calibration = y_development.iloc[
    calibration_split_index:
]


# -------------------------------------------------
# Show Dataset Sizes
# -------------------------------------------------

print("\n================================")
print("CALIBRATION MODEL COMPARISON")
print("================================")

print(
    "Total matches:",
    len(df)
)

print(
    "Training matches:",
    len(X_train)
)

print(
    "Calibration matches:",
    len(X_calibration)
)

print(
    "Final test matches:",
    len(X_test)
)


# -------------------------------------------------
# Original Logistic Regression
# -------------------------------------------------

original_model = LogisticRegression()

original_model.fit(
    X_train,
    y_train
)


# -------------------------------------------------
# Original Model Evaluation
# -------------------------------------------------

original_predictions = original_model.predict(
    X_test
)

original_probabilities = original_model.predict_proba(
    X_test
)[:, 1]

original_accuracy = accuracy_score(
    y_test,
    original_predictions
)

original_brier = brier_score_loss(
    y_test,
    original_probabilities
)


# -------------------------------------------------
# Sigmoid / Platt Calibration
#
# Base model is already fitted on X_train.
# FrozenEstimator prevents retraining.
# Calibration learns probability correction
# from X_calibration.
# -------------------------------------------------

sigmoid_base_model = LogisticRegression()

sigmoid_base_model.fit(
    X_train,
    y_train
)

sigmoid_model = CalibratedClassifierCV(
    FrozenEstimator(
        sigmoid_base_model
    ),
    method="sigmoid"
)

sigmoid_model.fit(
    X_calibration,
    y_calibration
)


# -------------------------------------------------
# Sigmoid Evaluation
# -------------------------------------------------

sigmoid_predictions = sigmoid_model.predict(
    X_test
)

sigmoid_probabilities = sigmoid_model.predict_proba(
    X_test
)[:, 1]

sigmoid_accuracy = accuracy_score(
    y_test,
    sigmoid_predictions
)

sigmoid_brier = brier_score_loss(
    y_test,
    sigmoid_probabilities
)


# -------------------------------------------------
# Isotonic Calibration
# -------------------------------------------------

isotonic_base_model = LogisticRegression()

isotonic_base_model.fit(
    X_train,
    y_train
)

isotonic_model = CalibratedClassifierCV(
    FrozenEstimator(
        isotonic_base_model
    ),
    method="isotonic"
)

isotonic_model.fit(
    X_calibration,
    y_calibration
)


# -------------------------------------------------
# Isotonic Evaluation
# -------------------------------------------------

isotonic_predictions = isotonic_model.predict(
    X_test
)

isotonic_probabilities = isotonic_model.predict_proba(
    X_test
)[:, 1]

isotonic_accuracy = accuracy_score(
    y_test,
    isotonic_predictions
)

isotonic_brier = brier_score_loss(
    y_test,
    isotonic_probabilities
)


# -------------------------------------------------
# Comparison Table
# -------------------------------------------------

comparison = pd.DataFrame([
    {
        "model": "Original Logistic Regression",
        "accuracy": original_accuracy,
        "brier_score": original_brier
    },
    {
        "model": "Sigmoid Calibrated",
        "accuracy": sigmoid_accuracy,
        "brier_score": sigmoid_brier
    },
    {
        "model": "Isotonic Calibrated",
        "accuracy": isotonic_accuracy,
        "brier_score": isotonic_brier
    }
])


comparison[
    "accuracy_percent"
] = (
    comparison["accuracy"]
    * 100
).round(2)

comparison[
    "brier_score"
] = comparison[
    "brier_score"
].round(4)


print(
    "\n--- MODEL COMPARISON ---"
)

print(
    comparison[
        [
            "model",
            "accuracy_percent",
            "brier_score"
        ]
    ].to_string(
        index=False
    )
)


# -------------------------------------------------
# Best Brier Score
# -------------------------------------------------

best_brier_row = comparison.loc[
    comparison[
        "brier_score"
    ].idxmin()
]


print(
    "\n--- BEST CALIBRATION RESULT ---"
)

print(
    "Best model by Brier Score:",
    best_brier_row["model"]
)

print(
    "Brier Score:",
    best_brier_row["brier_score"]
)

print(
    "Accuracy:",
    best_brier_row[
        "accuracy_percent"
    ],
    "%"
)


# -------------------------------------------------
# Improvement vs Original
# -------------------------------------------------

sigmoid_brier_improvement = (
    original_brier
    -
    sigmoid_brier
)

isotonic_brier_improvement = (
    original_brier
    -
    isotonic_brier
)


print(
    "\n--- BRIER SCORE IMPROVEMENT VS ORIGINAL ---"
)

print(
    "Sigmoid improvement:",
    round(
        sigmoid_brier_improvement,
        4
    )
)

print(
    "Isotonic improvement:",
    round(
        isotonic_brier_improvement,
        4
    )
)


# -------------------------------------------------
# Interpretation
# -------------------------------------------------

print(
    "\n--- INTERPRETATION ---"
)

print(
    "Lower Brier Score means better "
    "probability quality."
)

print(
    "Accuracy tells us how often the "
    "winner classification is correct."
)

print(
    "Calibration should improve probability "
    "quality without materially damaging accuracy."
)

print(
    "The final 20% test set was not used "
    "for model fitting or calibration."
)

print(
    "Sigmoid calibration is a smooth "
    "parametric probability correction."
)

print(
    "Isotonic calibration is more flexible "
    "but may overfit when calibration data "
    "is limited."
)