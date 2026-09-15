import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(
    r"K:\Python\Cricinfo_AI_Project"
)

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "matches_candidate_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "model"
    / "cricket_model_ablation_9feature.pkl"
)


FEATURES = [
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


print("========================================")
print("ABLATION TEST - 9 FEATURES")
print("========================================")

print(
    "\nPurpose:"
)

print(
    "Expanded dataset + Champion's original "
    "9-feature design"
)


# -------------------------------------------------
# Load candidate feature dataset
# -------------------------------------------------

df = pd.read_csv(
    DATA_FILE
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.sort_values(
    "date",
    kind="stable"
).reset_index(drop=True)


print(
    "\nTotal matches:",
    len(df)
)

print(
    "Features:",
    len(FEATURES)
)


# -------------------------------------------------
# Check missing values
# -------------------------------------------------

missing = df[
    FEATURES
].isna().sum().sum()


print(
    "Missing feature values:",
    missing
)


if missing > 0:

    raise ValueError(
        "Feature dataset contains missing values."
    )


# -------------------------------------------------
# Chronological 80/20 split
# -------------------------------------------------

split_index = int(
    len(df) * 0.80
)


train_df = df.iloc[
    :split_index
].copy()


test_df = df.iloc[
    split_index:
].copy()


X_train = train_df[
    FEATURES
]

y_train = train_df[
    "team1_won"
].astype(int)


X_test = test_df[
    FEATURES
]

y_test = test_df[
    "team1_won"
].astype(int)


print("\n========================================")
print("CHRONOLOGICAL SPLIT")
print("========================================")


print(
    "Training matches:",
    len(train_df)
)

print(
    "Testing matches:",
    len(test_df)
)


print(
    "Training period:",
    train_df["date"].min(),
    "to",
    train_df["date"].max()
)


print(
    "Testing period:",
    test_df["date"].min(),
    "to",
    test_df["date"].max()
)


# -------------------------------------------------
# Logistic Regression
# -------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


logistic_model.fit(
    X_train,
    y_train
)


logistic_predictions = (
    logistic_model.predict(
        X_test
    )
)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)


print("\n========================================")
print("LOGISTIC REGRESSION")
print("========================================")


print(
    "Accuracy:",
    round(
        logistic_accuracy,
        4
    )
)


# -------------------------------------------------
# Random Forest
# -------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


rf_model.fit(
    X_train,
    y_train
)


rf_predictions = (
    rf_model.predict(
        X_test
    )
)


rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)


print("\n========================================")
print("RANDOM FOREST")
print("========================================")


print(
    "Accuracy:",
    round(
        rf_accuracy,
        4
    )
)


# -------------------------------------------------
# Select best
# -------------------------------------------------

if logistic_accuracy >= rf_accuracy:

    selected_model = logistic_model

    selected_name = (
        "LogisticRegression"
    )

    selected_accuracy = (
        logistic_accuracy
    )

else:

    selected_model = rf_model

    selected_name = (
        "RandomForest"
    )

    selected_accuracy = (
        rf_accuracy
    )


print("\n========================================")
print("MODEL SELECTION")
print("========================================")


print(
    "Selected model:",
    selected_name
)


print(
    "Accuracy:",
    round(
        selected_accuracy,
        4
    )
)


# -------------------------------------------------
# Feature coefficients
# -------------------------------------------------

if selected_name == "LogisticRegression":

    coefficients = pd.DataFrame({

        "feature":
            FEATURES,

        "coefficient":
            selected_model.coef_[0]
    })


    coefficients[
        "absolute_influence"
    ] = coefficients[
        "coefficient"
    ].abs()


    coefficients = (
        coefficients
        .sort_values(
            "absolute_influence",
            ascending=False
        )
    )


    print("\n========================================")
    print("FEATURE INFLUENCE")
    print("========================================")


    print(
        coefficients[
            [
                "feature",
                "coefficient"
            ]
        ].to_string(
            index=False
        )
    )


# -------------------------------------------------
# Afghanistan test performance
# -------------------------------------------------

afghanistan_mask = (
    (test_df["team1"] == "Afghanistan")
    |
    (test_df["team2"] == "Afghanistan")
)


afghanistan_count = int(
    afghanistan_mask.sum()
)


print("\n========================================")
print("AFGHANISTAN TEST PERFORMANCE")
print("========================================")


print(
    "Afghanistan matches in test set:",
    afghanistan_count
)


afghanistan_accuracy = None


if afghanistan_count > 0:

    afghanistan_predictions = (
        selected_model.predict(
            test_df.loc[
                afghanistan_mask,
                FEATURES
            ]
        )
    )


    afghanistan_accuracy = accuracy_score(

        test_df.loc[
            afghanistan_mask,
            "team1_won"
        ].astype(int),

        afghanistan_predictions
    )


    print(
        "Afghanistan accuracy:",
        round(
            afghanistan_accuracy,
            4
        )
    )


# -------------------------------------------------
# Save model package
# -------------------------------------------------

package = {

    "model":
        selected_model,

    "model_name":
        selected_name,

    "version":
        "ablation_9feature",

    "features":
        FEATURES,

    "accuracy":
        selected_accuracy,

    "training_rows":
        len(train_df),

    "testing_rows":
        len(test_df),

    "total_rows":
        len(df),

    "training_period":
        (
            str(train_df["date"].min()),
            str(train_df["date"].max())
        ),

    "testing_period":
        (
            str(test_df["date"].min()),
            str(test_df["date"].max())
        ),

    "afghanistan_test_matches":
        afghanistan_count,

    "afghanistan_accuracy":
        afghanistan_accuracy
}


joblib.dump(
    package,
    MODEL_FILE
)


print("\n========================================")
print("ABLATION MODEL SAVED")
print("========================================")


print(
    "Model:",
    MODEL_FILE
)


print(
    "Algorithm:",
    selected_name
)


print(
    "Accuracy:",
    round(
        selected_accuracy,
        4
    )
)


print(
    "\nChampion v1.0 was NOT modified."
)

print(
    "Challenger v1.1 was NOT modified."
)