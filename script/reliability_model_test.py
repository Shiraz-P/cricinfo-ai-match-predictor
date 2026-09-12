import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix
)


# -------------------------------------------------
# Load Feature Dataset
# -------------------------------------------------

feature_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\matches_features.csv"
)

df = pd.read_csv(feature_file)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# -------------------------------------------------
# Load Reliability Dataset
# -------------------------------------------------

reliability_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\feature_reliability_analysis.csv"
)

reliability_df = pd.read_csv(
    reliability_file
)

reliability_df["date"] = pd.to_datetime(
    reliability_df["date"]
)

reliability_df = reliability_df.sort_values(
    "date"
).reset_index(
    drop=True
)


# -------------------------------------------------
# Safety Check
# -------------------------------------------------

if len(df) != len(reliability_df):

    raise ValueError(
        "Dataset row counts do not match."
    )


print("\n================================")
print("RELIABILITY-AWARE MODEL TEST")
print("================================")

print(
    "Total matches:",
    len(df)
)


# -------------------------------------------------
# Baseline 9 Features
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
# Add Reliability Sample Counts
# -------------------------------------------------

df[
    "team1_historical_sample_size"
] = reliability_df[
    "team1_historical_sample_size"
]

df[
    "team2_historical_sample_size"
] = reliability_df[
    "team2_historical_sample_size"
]

# H2H count is the same for Team 1 and Team 2
df[
    "h2h_sample_size"
] = reliability_df[
    "team1_h2h_sample_size"
]

df[
    "team1_venue_sample_size"
] = reliability_df[
    "team1_venue_sample_size"
]

df[
    "team2_venue_sample_size"
] = reliability_df[
    "team2_venue_sample_size"
]


# -------------------------------------------------
# Log Transform Sample Counts
#
# log1p(x) = log(1 + x)
#
# Example:
# 0   -> 0
# 1   -> 0.69
# 10  -> 2.40
# 100 -> 4.62
#
# This compresses very large count values.
# -------------------------------------------------

df[
    "team1_historical_sample_log"
] = np.log1p(
    df[
        "team1_historical_sample_size"
    ]
)

df[
    "team2_historical_sample_log"
] = np.log1p(
    df[
        "team2_historical_sample_size"
    ]
)

df[
    "h2h_sample_log"
] = np.log1p(
    df[
        "h2h_sample_size"
    ]
)

df[
    "team1_venue_sample_log"
] = np.log1p(
    df[
        "team1_venue_sample_size"
    ]
)

df[
    "team2_venue_sample_log"
] = np.log1p(
    df[
        "team2_venue_sample_size"
    ]
)


# -------------------------------------------------
# Reliability Features
# -------------------------------------------------

reliability_features = [
    "team1_historical_sample_log",
    "team2_historical_sample_log",
    "h2h_sample_log",
    "team1_venue_sample_log",
    "team2_venue_sample_log"
]


# -------------------------------------------------
# 14 Feature Model
# -------------------------------------------------

enhanced_features = (
    baseline_features
    +
    reliability_features
)


# -------------------------------------------------
# Target
# -------------------------------------------------

y = df[
    "team1_won"
]


# -------------------------------------------------
# Chronological 80/20 Split
# -------------------------------------------------

split_index = int(
    len(df) * 0.8
)

print(
    "\nTraining matches:",
    split_index
)

print(
    "Testing matches:",
    len(df) - split_index
)

print(
    "Training period:",
    df.iloc[
        :split_index
    ]["date"].min(),
    "to",
    df.iloc[
        :split_index
    ]["date"].max()
)

print(
    "Testing period:",
    df.iloc[
        split_index:
    ]["date"].min(),
    "to",
    df.iloc[
        split_index:
    ]["date"].max()
)


# =================================================
# MODEL 1
# BASELINE 9-FEATURE MODEL
# =================================================

print(
    "\n================================"
)

print(
    "MODEL 1: BASELINE 9 FEATURES"
)

print(
    "================================"
)


X_baseline = df[
    baseline_features
]

X_baseline_train = X_baseline.iloc[
    :split_index
]

X_baseline_test = X_baseline.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


baseline_model = LogisticRegression(
    max_iter=1000
)

baseline_model.fit(
    X_baseline_train,
    y_train
)


baseline_predictions = baseline_model.predict(
    X_baseline_test
)

baseline_probabilities = (
    baseline_model.predict_proba(
        X_baseline_test
    )[:, 1]
)


baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_brier = brier_score_loss(
    y_test,
    baseline_probabilities
)


print(
    "Features:",
    len(baseline_features)
)

print(
    "Accuracy:",
    round(
        baseline_accuracy * 100,
        2
    ),
    "%"
)

print(
    "Brier Score:",
    round(
        baseline_brier,
        4
    )
)


# =================================================
# MODEL 2
# RELIABILITY-AWARE 14-FEATURE MODEL
# =================================================

print(
    "\n================================"
)

print(
    "MODEL 2: RELIABILITY-AWARE 14 FEATURES"
)

print(
    "================================"
)


X_enhanced = df[
    enhanced_features
]

X_enhanced_train = X_enhanced.iloc[
    :split_index
]

X_enhanced_test = X_enhanced.iloc[
    split_index:
]


enhanced_model = LogisticRegression(
    max_iter=1000
)

enhanced_model.fit(
    X_enhanced_train,
    y_train
)


enhanced_predictions = enhanced_model.predict(
    X_enhanced_test
)

enhanced_probabilities = (
    enhanced_model.predict_proba(
        X_enhanced_test
    )[:, 1]
)


enhanced_accuracy = accuracy_score(
    y_test,
    enhanced_predictions
)

enhanced_brier = brier_score_loss(
    y_test,
    enhanced_probabilities
)


print(
    "Features:",
    len(enhanced_features)
)

print(
    "Accuracy:",
    round(
        enhanced_accuracy * 100,
        2
    ),
    "%"
)

print(
    "Brier Score:",
    round(
        enhanced_brier,
        4
    )
)


# -------------------------------------------------
# Accuracy Difference
# -------------------------------------------------

accuracy_difference = (
    enhanced_accuracy
    -
    baseline_accuracy
)


# -------------------------------------------------
# Brier Difference
#
# Positive improvement means enhanced model
# has LOWER / better Brier score.
# -------------------------------------------------

brier_improvement = (
    baseline_brier
    -
    enhanced_brier
)


# -------------------------------------------------
# Final Comparison
# -------------------------------------------------

print(
    "\n================================"
)

print(
    "FINAL MODEL COMPARISON"
)

print(
    "================================"
)


comparison = pd.DataFrame([
    {
        "model":
            "Baseline 9 Features",

        "features":
            len(
                baseline_features
            ),

        "accuracy":
            baseline_accuracy
            * 100,

        "brier_score":
            baseline_brier
    },

    {
        "model":
            "Reliability-Aware 14 Features",

        "features":
            len(
                enhanced_features
            ),

        "accuracy":
            enhanced_accuracy
            * 100,

        "brier_score":
            enhanced_brier
    }
])


comparison[
    "accuracy"
] = comparison[
    "accuracy"
].round(
    2
)

comparison[
    "brier_score"
] = comparison[
    "brier_score"
].round(
    4
)


print(
    comparison.to_string(
        index=False
    )
)


print(
    "\nAccuracy difference:",
    round(
        accuracy_difference
        * 100,
        2
    ),
    "percentage points"
)

print(
    "Brier improvement:",
    round(
        brier_improvement,
        4
    )
)


# -------------------------------------------------
# Determine Current Winner
# -------------------------------------------------

print(
    "\n--- INTERPRETATION ---"
)


if (
    enhanced_accuracy
    >
    baseline_accuracy
):

    print(
        "Reliability-aware model has "
        "higher classification accuracy."
    )

elif (
    enhanced_accuracy
    <
    baseline_accuracy
):

    print(
        "Baseline model has higher "
        "classification accuracy."
    )

else:

    print(
        "Both models have the same accuracy."
    )


if (
    enhanced_brier
    <
    baseline_brier
):

    print(
        "Reliability-aware model has "
        "better probability quality."
    )

elif (
    enhanced_brier
    >
    baseline_brier
):

    print(
        "Baseline model has better "
        "probability quality."
    )

else:

    print(
        "Both models have the same "
        "Brier Score."
    )


# -------------------------------------------------
# Reliability Feature Coefficients
# -------------------------------------------------

coefficients = pd.DataFrame({
    "feature":
        enhanced_features,

    "coefficient":
        enhanced_model.coef_[0]
})


reliability_coefficients = coefficients[
    coefficients[
        "feature"
    ].isin(
        reliability_features
    )
].copy()


reliability_coefficients[
    "absolute_coefficient"
] = (
    reliability_coefficients[
        "coefficient"
    ].abs()
)


reliability_coefficients = (
    reliability_coefficients
    .sort_values(
        by="absolute_coefficient",
        ascending=False
    )
)


print(
    "\n--- RELIABILITY FEATURE COEFFICIENTS ---"
)

print(
    reliability_coefficients
    .round(4)
    .to_string(
        index=False
    )
)


# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

print(
    "\n--- ENHANCED MODEL CONFUSION MATRIX ---"
)

print(
    confusion_matrix(
        y_test,
        enhanced_predictions
    )
)


# -------------------------------------------------
# Classification Report
# -------------------------------------------------

print(
    "\n--- ENHANCED MODEL CLASSIFICATION REPORT ---"
)

print(
    classification_report(
        y_test,
        enhanced_predictions
    )
)


# -------------------------------------------------
# Important Note
# -------------------------------------------------

print(
    "\nImportant:"
)

print(
    "This is an experiment only."
)

print(
    "The production cricket_model.pkl "
    "has NOT been changed."
)

print(
    "We will only consider adding reliability "
    "features if they improve validation results."
)