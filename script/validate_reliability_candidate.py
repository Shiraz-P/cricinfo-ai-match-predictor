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
# Baseline 9 features
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
# Add historical reliability
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

X_baseline = df[
    baseline_features
]

X_candidate = df[
    candidate_features
]

y = df[
    "team1_won"
]


# -------------------------------------------------
# Rolling folds
# -------------------------------------------------

number_of_folds = 5

fold_size = (
    len(df)
    //
    (number_of_folds + 1)
)

results = []


print("\n================================")
print("RELIABILITY CANDIDATE VALIDATION")
print("================================")

print(
    "Total matches:",
    len(df)
)


# -------------------------------------------------
# Rolling validation
# -------------------------------------------------

for fold in range(
    1,
    number_of_folds + 1
):

    train_end = (
        fold_size
        *
        fold
    )

    test_end = (
        train_end
        +
        fold_size
    )

    if test_end > len(df):
        test_end = len(df)


    y_train = y.iloc[
        :train_end
    ]

    y_test = y.iloc[
        train_end:test_end
    ]


    # ---------------------------------------------
    # Baseline model
    # ---------------------------------------------

    baseline_model = LogisticRegression(
        max_iter=1000
    )

    baseline_model.fit(
        X_baseline.iloc[
            :train_end
        ],
        y_train
    )

    baseline_pred = baseline_model.predict(
        X_baseline.iloc[
            train_end:test_end
        ]
    )

    baseline_prob = baseline_model.predict_proba(
        X_baseline.iloc[
            train_end:test_end
        ]
    )[:, 1]


    baseline_accuracy = accuracy_score(
        y_test,
        baseline_pred
    )

    baseline_brier = brier_score_loss(
        y_test,
        baseline_prob
    )


    # ---------------------------------------------
    # Candidate 11-feature model
    # ---------------------------------------------

    candidate_model = LogisticRegression(
        max_iter=1000
    )

    candidate_model.fit(
        X_candidate.iloc[
            :train_end
        ],
        y_train
    )

    candidate_pred = candidate_model.predict(
        X_candidate.iloc[
            train_end:test_end
        ]
    )

    candidate_prob = candidate_model.predict_proba(
        X_candidate.iloc[
            train_end:test_end
        ]
    )[:, 1]


    candidate_accuracy = accuracy_score(
        y_test,
        candidate_pred
    )

    candidate_brier = brier_score_loss(
        y_test,
        candidate_prob
    )


    # ---------------------------------------------
    # Store result
    # ---------------------------------------------

    results.append({
        "fold": fold,
        "training_matches":
            train_end,

        "testing_matches":
            len(y_test),

        "baseline_accuracy":
            baseline_accuracy,

        "baseline_brier":
            baseline_brier,

        "candidate_accuracy":
            candidate_accuracy,

        "candidate_brier":
            candidate_brier
    })


    print("\n-----------------------------")
    print("Fold:", fold)

    print(
        "Training matches:",
        train_end
    )

    print(
        "Testing matches:",
        len(y_test)
    )

    print(
        "Testing period:",
        df.iloc[
            train_end:test_end
        ]["date"].min(),
        "to",
        df.iloc[
            train_end:test_end
        ]["date"].max()
    )

    print(
        "Baseline Accuracy:",
        round(
            baseline_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Candidate Accuracy:",
        round(
            candidate_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Baseline Brier:",
        round(
            baseline_brier,
            4
        )
    )

    print(
        "Candidate Brier:",
        round(
            candidate_brier,
            4
        )
    )


# -------------------------------------------------
# Summary
# -------------------------------------------------

results_df = pd.DataFrame(
    results
)


print("\n================================")
print("ROLLING VALIDATION SUMMARY")
print("================================")


baseline_avg_accuracy = (
    results_df[
        "baseline_accuracy"
    ].mean()
)

candidate_avg_accuracy = (
    results_df[
        "candidate_accuracy"
    ].mean()
)

baseline_avg_brier = (
    results_df[
        "baseline_brier"
    ].mean()
)

candidate_avg_brier = (
    results_df[
        "candidate_brier"
    ].mean()
)


print(
    "Baseline Average Accuracy:",
    round(
        baseline_avg_accuracy * 100,
        2
    ),
    "%"
)

print(
    "Candidate Average Accuracy:",
    round(
        candidate_avg_accuracy * 100,
        2
    ),
    "%"
)

print(
    "\nBaseline Average Brier:",
    round(
        baseline_avg_brier,
        4
    )
)

print(
    "Candidate Average Brier:",
    round(
        candidate_avg_brier,
        4
    )
)


# -------------------------------------------------
# Improvement counts
# -------------------------------------------------

accuracy_wins = (
    results_df[
        "candidate_accuracy"
    ]
    >
    results_df[
        "baseline_accuracy"
    ]
).sum()

brier_wins = (
    results_df[
        "candidate_brier"
    ]
    <
    results_df[
        "baseline_brier"
    ]
).sum()


print(
    "\nCandidate accuracy wins:",
    accuracy_wins,
    "out of",
    len(results_df)
)

print(
    "Candidate Brier wins:",
    brier_wins,
    "out of",
    len(results_df)
)


# -------------------------------------------------
# Recommendation
# -------------------------------------------------

print(
    "\n--- RECOMMENDATION ---"
)

if (
    candidate_avg_accuracy
    >
    baseline_avg_accuracy
    and
    candidate_avg_brier
    <
    baseline_avg_brier
):

    print(
        "Candidate improves both average "
        "accuracy and probability quality."
    )

    print(
        "Candidate is suitable for further "
        "promotion testing."
    )

else:

    print(
        "Candidate does not consistently "
        "improve both metrics."
    )

    print(
        "Keep current production model."
    )


print(
    "\nImportant:"
)

print(
    "This script does not modify cricket_model.pkl."
)