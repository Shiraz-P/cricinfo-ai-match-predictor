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
# Time-based folds
# -------------------------------------------------

number_of_folds = 5

fold_size = len(df) // (number_of_folds + 1)

results = []


print("\n================================")
print("TIME-BASED CALIBRATION VALIDATION")
print("================================")


# -------------------------------------------------
# Rolling validation
# -------------------------------------------------

for fold in range(1, number_of_folds + 1):

    train_end = fold_size * fold
    test_end = train_end + fold_size

    if test_end > len(df):
        test_end = len(df)

    X_development = X.iloc[:train_end]
    y_development = y.iloc[:train_end]

    X_test = X.iloc[
        train_end:test_end
    ]

    y_test = y.iloc[
        train_end:test_end
    ]

    if len(X_test) == 0:
        continue


    # ---------------------------------------------
    # Split development data:
    # 80% base training
    # 20% calibration
    # ---------------------------------------------

    calibration_split = int(
        len(X_development) * 0.8
    )

    X_train = X_development.iloc[
        :calibration_split
    ]

    y_train = y_development.iloc[
        :calibration_split
    ]

    X_calibration = X_development.iloc[
        calibration_split:
    ]

    y_calibration = y_development.iloc[
        calibration_split:
    ]


    # ---------------------------------------------
    # Original Logistic Regression
    # ---------------------------------------------

    original_model = LogisticRegression()

    original_model.fit(
        X_train,
        y_train
    )

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


    # ---------------------------------------------
    # Sigmoid Calibration
    # ---------------------------------------------

    sigmoid_base = LogisticRegression()

    sigmoid_base.fit(
        X_train,
        y_train
    )

    sigmoid_model = CalibratedClassifierCV(
        FrozenEstimator(
            sigmoid_base
        ),
        method="sigmoid"
    )

    sigmoid_model.fit(
        X_calibration,
        y_calibration
    )

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


    # ---------------------------------------------
    # Isotonic Calibration
    # ---------------------------------------------

    isotonic_base = LogisticRegression()

    isotonic_base.fit(
        X_train,
        y_train
    )

    isotonic_model = CalibratedClassifierCV(
        FrozenEstimator(
            isotonic_base
        ),
        method="isotonic"
    )

    isotonic_model.fit(
        X_calibration,
        y_calibration
    )

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


    # ---------------------------------------------
    # Store Fold Results
    # ---------------------------------------------

    results.append({
        "fold": fold,
        "development_matches":
            len(X_development),

        "training_matches":
            len(X_train),

        "calibration_matches":
            len(X_calibration),

        "test_matches":
            len(X_test),

        "original_accuracy":
            original_accuracy,

        "original_brier":
            original_brier,

        "sigmoid_accuracy":
            sigmoid_accuracy,

        "sigmoid_brier":
            sigmoid_brier,

        "isotonic_accuracy":
            isotonic_accuracy,

        "isotonic_brier":
            isotonic_brier
    })


    # ---------------------------------------------
    # Fold output
    # ---------------------------------------------

    print("\n-----------------------------")
    print("Fold:", fold)

    print(
        "Development matches:",
        len(X_development)
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
        "Testing matches:",
        len(X_test)
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
        "Original Accuracy:",
        round(
            original_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Original Brier:",
        round(
            original_brier,
            4
        )
    )

    print(
        "Sigmoid Accuracy:",
        round(
            sigmoid_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Sigmoid Brier:",
        round(
            sigmoid_brier,
            4
        )
    )

    print(
        "Isotonic Accuracy:",
        round(
            isotonic_accuracy * 100,
            2
        ),
        "%"
    )

    print(
        "Isotonic Brier:",
        round(
            isotonic_brier,
            4
        )
    )


# -------------------------------------------------
# Final Results DataFrame
# -------------------------------------------------

results_df = pd.DataFrame(
    results
)


# -------------------------------------------------
# Average Results
# -------------------------------------------------

print("\n================================")
print("CALIBRATION VALIDATION SUMMARY")
print("================================")


print(
    "Average Original Accuracy:",
    round(
        results_df[
            "original_accuracy"
        ].mean() * 100,
        2
    ),
    "%"
)

print(
    "Average Original Brier:",
    round(
        results_df[
            "original_brier"
        ].mean(),
        4
    )
)


print(
    "\nAverage Sigmoid Accuracy:",
    round(
        results_df[
            "sigmoid_accuracy"
        ].mean() * 100,
        2
    ),
    "%"
)

print(
    "Average Sigmoid Brier:",
    round(
        results_df[
            "sigmoid_brier"
        ].mean(),
        4
    )
)


print(
    "\nAverage Isotonic Accuracy:",
    round(
        results_df[
            "isotonic_accuracy"
        ].mean() * 100,
        2
    ),
    "%"
)

print(
    "Average Isotonic Brier:",
    round(
        results_df[
            "isotonic_brier"
        ].mean(),
        4
    )
)


# -------------------------------------------------
# Determine Best Average Brier Score
# -------------------------------------------------

average_briers = {
    "Original Logistic Regression":
        results_df[
            "original_brier"
        ].mean(),

    "Sigmoid Calibrated":
        results_df[
            "sigmoid_brier"
        ].mean(),

    "Isotonic Calibrated":
        results_df[
            "isotonic_brier"
        ].mean()
}

best_model = min(
    average_briers,
    key=average_briers.get
)


print(
    "\nBest model by average Brier Score:",
    best_model
)


# -------------------------------------------------
# Important interpretation
# -------------------------------------------------

print(
    "\nImportant:"
)

print(
    "Lower average Brier Score indicates "
    "more reliable probability predictions."
)

print(
    "A calibration method should perform "
    "consistently across multiple time periods."
)

print(
    "Do not promote a calibrated model based "
    "on only one test period."
)