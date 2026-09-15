import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "matches_candidate_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "model"

MODEL_FILE = (
    MODEL_DIR
    / "cricket_model_challenger_v11.pkl"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# -------------------------------------------------
# Load feature dataset
# -------------------------------------------------

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.sort_values(
    "date"
).reset_index(drop=True)

print("========================================")
print("TRAIN CHALLENGER v1.1")
print("========================================")

print("\nTotal matches:", len(df))

# -------------------------------------------------
# Challenger features
# -------------------------------------------------

features = [
    "toss_winner_is_team1",
    "toss_data_known",
    "team1_historical_win_rate",
    "team2_historical_win_rate",
    "team1_recent_win_rate",
    "team2_recent_win_rate",
    "team1_h2h_win_rate",
    "team2_h2h_win_rate",
    "team1_venue_win_rate",
    "team2_venue_win_rate"
]

target = "team1_won"

X = df[features]
y = df[target]

print("Features:", len(features))

# -------------------------------------------------
# Chronological 80/20 split
# -------------------------------------------------

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print("\n========================================")
print("CHRONOLOGICAL SPLIT")
print("========================================")

print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))

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

print("\n========================================")
print("LOGISTIC REGRESSION")
print("========================================")

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_predictions = (
    logistic_model.predict(X_test)
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print(
    "Accuracy:",
    round(logistic_accuracy, 4)
)

# -------------------------------------------------
# Random Forest
# -------------------------------------------------

print("\n========================================")
print("RANDOM FOREST")
print("========================================")

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
    rf_model.predict(X_test)
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print(
    "Accuracy:",
    round(rf_accuracy, 4)
)

# -------------------------------------------------
# Select best Challenger model
# -------------------------------------------------

print("\n========================================")
print("MODEL SELECTION")
print("========================================")

if logistic_accuracy >= rf_accuracy:

    best_model = logistic_model
    best_name = "LogisticRegression"
    best_accuracy = logistic_accuracy
    best_predictions = logistic_predictions

else:

    best_model = rf_model
    best_name = "RandomForestClassifier"
    best_accuracy = rf_accuracy
    best_predictions = rf_predictions

print(
    "Selected Challenger model:",
    best_name
)

print(
    "Challenger accuracy:",
    round(best_accuracy, 4)
)

# -------------------------------------------------
# Classification report
# -------------------------------------------------

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        best_predictions,
        digits=4,
        zero_division=0
    )
)

# -------------------------------------------------
# Confusion matrix
# -------------------------------------------------

print("========================================")
print("CONFUSION MATRIX")
print("========================================")

print(
    confusion_matrix(
        y_test,
        best_predictions
    )
)

# -------------------------------------------------
# Afghanistan test-set performance
# -------------------------------------------------

afg_mask = (
    (test_df["team1"] == "Afghanistan")
    |
    (test_df["team2"] == "Afghanistan")
)

afg_count = int(
    afg_mask.sum()
)

print("\n========================================")
print("AFGHANISTAN TEST PERFORMANCE")
print("========================================")

print(
    "Afghanistan matches in test set:",
    afg_count
)

if afg_count > 0:

    afg_actual = y_test[
        afg_mask.values
    ]

    afg_predictions = (
        best_predictions[
            afg_mask.values
        ]
    )

    afg_accuracy = accuracy_score(
        afg_actual,
        afg_predictions
    )

    print(
        "Afghanistan accuracy:",
        round(afg_accuracy, 4)
    )

else:

    afg_accuracy = None

    print(
        "No Afghanistan matches "
        "in test period."
    )

# -------------------------------------------------
# Feature importance / coefficients
# -------------------------------------------------

print("\n========================================")
print("FEATURE INFLUENCE")
print("========================================")

if best_name == "RandomForestClassifier":

    importance = pd.DataFrame({
        "feature": features,
        "value": best_model.feature_importances_
    })

else:

    importance = pd.DataFrame({
        "feature": features,
        "value": best_model.coef_[0]
    })

importance["absolute_value"] = (
    importance["value"].abs()
)

importance = importance.sort_values(
    "absolute_value",
    ascending=False
)

print(
    importance[
        ["feature", "value"]
    ].to_string(index=False)
)

# -------------------------------------------------
# Save model package
#
# We save model + metadata together so later
# prediction scripts know exactly which features
# this Challenger expects.
# -------------------------------------------------

model_package = {

    "model": best_model,

    "model_name": best_name,

    "version": "1.1",

    "features": features,

    "accuracy": float(best_accuracy),

    "training_rows": len(X_train),

    "testing_rows": len(X_test),

    "total_rows": len(df),

    "train_start": str(
        train_df["date"].min()
    ),

    "train_end": str(
        train_df["date"].max()
    ),

    "test_start": str(
        test_df["date"].min()
    ),

    "test_end": str(
        test_df["date"].max()
    ),

    "afghanistan_test_matches": afg_count,

    "afghanistan_test_accuracy": (
        None
        if afg_accuracy is None
        else float(afg_accuracy)
    )
}

joblib.dump(
    model_package,
    MODEL_FILE
)

print("\n========================================")
print("CHALLENGER v1.1 SAVED")
print("========================================")

print(
    "Model:",
    MODEL_FILE
)

print(
    "Algorithm:",
    best_name
)

print(
    "Accuracy:",
    round(best_accuracy, 4)
)

print(
    "\nChampion v1.0 was NOT modified."
)