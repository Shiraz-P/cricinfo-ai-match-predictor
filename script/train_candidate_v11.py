import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss
)


# -------------------------------------------------
# Load datasets
# -------------------------------------------------

feature_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\matches_features.csv"
)

reliability_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\feature_reliability_analysis.csv"
)

df = pd.read_csv(feature_file)
reliability_df = pd.read_csv(reliability_file)

df["date"] = pd.to_datetime(df["date"])
reliability_df["date"] = pd.to_datetime(
    reliability_df["date"]
)

df = df.sort_values(
    "date"
).reset_index(drop=True)

reliability_df = reliability_df.sort_values(
    "date"
).reset_index(drop=True)


# -------------------------------------------------
# Safety check
# -------------------------------------------------

if len(df) != len(reliability_df):

    raise ValueError(
        "Dataset row counts do not match."
    )


# -------------------------------------------------
# Original 9 features
# -------------------------------------------------

baseline_features = [
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


# -------------------------------------------------
# Add historical sample reliability
# -------------------------------------------------

df[
    "team1_historical_sample_log"
] = np.log1p(
    reliability_df[
        "team1_historical_sample_size"
    ]
)

df[
    "team2_historical_sample_log"
] = np.log1p(
    reliability_df[
        "team2_historical_sample_size"
    ]
)


# -------------------------------------------------
# Candidate 11 features
# -------------------------------------------------

candidate_features = (
    baseline_features
    +
    [
        "team1_historical_sample_log",
        "team2_historical_sample_log"
    ]
)


# -------------------------------------------------
# Data
# -------------------------------------------------

X = df[
    candidate_features
]

y = df[
    "team1_won"
]


# -------------------------------------------------
# Chronological 80/20 split
# -------------------------------------------------

split_index = int(
    len(df) * 0.8
)

X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


# -------------------------------------------------
# Train candidate model
# -------------------------------------------------

candidate_model = LogisticRegression(
    max_iter=1000
)

candidate_model.fit(
    X_train,
    y_train
)


# -------------------------------------------------
# Evaluate candidate
# -------------------------------------------------

predictions = candidate_model.predict(
    X_test
)

probabilities = candidate_model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

brier = brier_score_loss(
    y_test,
    probabilities
)


print("\n================================")
print("MODEL V1.1 CANDIDATE TRAINING")
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
    "Testing matches:",
    len(X_test)
)

print(
    "Feature count:",
    len(candidate_features)
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
    "Brier Score:",
    round(
        brier,
        4
    )
)


# -------------------------------------------------
# Candidate metadata
# -------------------------------------------------

model_package = {

    "model":
        candidate_model,

    "features":
        candidate_features,

    "feature_count":
        len(candidate_features),

    "model_version":
        "1.1-candidate",

    "model_type":
        "Logistic Regression",

    "test_accuracy":
        round(
            accuracy * 100,
            2
        ),

    "brier_score":
        round(
            brier,
            4
        ),

    "training_matches":
        len(X_train),

    "testing_matches":
        len(X_test),

    "reliability_features":
        [
            "team1_historical_sample_log",
            "team2_historical_sample_log"
        ],

    "status":
        "CHALLENGER"
}


# -------------------------------------------------
# Save separately from production v1.0
# -------------------------------------------------

candidate_path = (
    r"K:\Python\Cricinfo_AI_Project\model"
    r"\cricket_model_v1_1_candidate.pkl"
)

joblib.dump(
    model_package,
    candidate_path
)


print("\n--- CANDIDATE MODEL SAVED ---")

print(
    "Model location:",
    candidate_path
)

print(
    "Model version:",
    model_package[
        "model_version"
    ]
)

print(
    "Status:",
    model_package[
        "status"
    ]
)

print(
    "\nIMPORTANT:"
)

print(
    "Production cricket_model.pkl "
    "has NOT been modified."
)