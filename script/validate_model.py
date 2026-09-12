import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

file_path = r"K:\Python\Cricinfo_AI_Project\data\matches_features.csv"

df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# -------------------------------------------------
# Select 9 Features
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


print("\n--- TIME-BASED MODEL VALIDATION ---")

print("Total matches:", len(df))
print("Features:", len(features))


# -------------------------------------------------
# Create Time-Series Splits
# -------------------------------------------------

tscv = TimeSeriesSplit(n_splits=5)

fold_accuracies = []

fold_number = 1


# -------------------------------------------------
# Train and Test Each Fold
# -------------------------------------------------

for train_index, test_index in tscv.split(X):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = LogisticRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    fold_accuracies.append(accuracy)

    train_start_date = df.iloc[train_index[0]]["date"]
    train_end_date = df.iloc[train_index[-1]]["date"]

    test_start_date = df.iloc[test_index[0]]["date"]
    test_end_date = df.iloc[test_index[-1]]["date"]

    print("\n-----------------------------")
    print("Fold:", fold_number)

    print(
        "Training matches:",
        len(train_index)
    )

    print(
        "Testing matches:",
        len(test_index)
    )

    print(
        "Training period:",
        train_start_date.date(),
        "to",
        train_end_date.date()
    )

    print(
        "Testing period:",
        test_start_date.date(),
        "to",
        test_end_date.date()
    )

    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    fold_number += 1


# -------------------------------------------------
# Overall Validation Result
# -------------------------------------------------

average_accuracy = sum(fold_accuracies) / len(fold_accuracies)

best_accuracy = max(fold_accuracies)

worst_accuracy = min(fold_accuracies)


print("\n================================")
print("ROLLING VALIDATION SUMMARY")
print("================================")

for i, score in enumerate(fold_accuracies, start=1):

    print(
        "Fold",
        i,
        ":",
        round(score * 100, 2),
        "%"
    )


print(
    "\nAverage Accuracy:",
    round(average_accuracy * 100, 2),
    "%"
)

print(
    "Best Fold Accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)

print(
    "Worst Fold Accuracy:",
    round(worst_accuracy * 100, 2),
    "%"
)