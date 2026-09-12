import pandas as pd

from sklearn.linear_model import LogisticRegression
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
# Make Predictions
# -------------------------------------------------

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)


# -------------------------------------------------
# Create Error Analysis Dataset
# -------------------------------------------------

results = df.iloc[split_index:].copy()

results["actual"] = y_test.values

results["predicted"] = predictions

results["team1_probability"] = probabilities[:, 1]

results["team2_probability"] = probabilities[:, 0]

results["correct_prediction"] = (
    results["actual"]
    ==
    results["predicted"]
)


# -------------------------------------------------
# Overall Results
# -------------------------------------------------

print("\n--- ERROR ANALYSIS ---")

print(
    "Total test matches:",
    len(results)
)

correct = results[
    "correct_prediction"
].sum()

incorrect = (
    len(results)
    -
    correct
)

print(
    "Correct predictions:",
    correct
)

print(
    "Incorrect predictions:",
    incorrect
)

print(
    "Accuracy:",
    round(
        correct / len(results) * 100,
        2
    ),
    "%"
)


# -------------------------------------------------
# Show Incorrect Predictions
# -------------------------------------------------

errors = results[
    results["correct_prediction"] == False
].copy()

print(
    "\n--- FIRST 20 WRONG PREDICTIONS ---"
)

print(
    errors[
        [
            "date",
            "team1",
            "team2",
            "venue",
            "winner",
            "team1_probability",
            "team2_probability"
        ]
    ].head(20).to_string(
        index=False
    )
)


# -------------------------------------------------
# Analyze Similar Strength Teams
# -------------------------------------------------

results["historical_difference"] = abs(
    results["team1_historical_win_rate"]
    -
    results["team2_historical_win_rate"]
)

close_matches = results[
    results["historical_difference"] <= 0.10
]

if len(close_matches) > 0:

    close_accuracy = accuracy_score(
        close_matches["actual"],
        close_matches["predicted"]
    )

    print(
        "\n--- SIMILAR STRENGTH TEAMS ---"
    )

    print(
        "Matches:",
        len(close_matches)
    )

    print(
        "Accuracy:",
        round(
            close_accuracy * 100,
            2
        ),
        "%"
    )


# -------------------------------------------------
# Analyze Toss Impact
# -------------------------------------------------

team1_won_toss = results[
    results["toss_winner_is_team1"] == 1
]

team2_won_toss = results[
    results["toss_winner_is_team1"] == 0
]

print(
    "\n--- TOSS ANALYSIS ---"
)

if len(team1_won_toss) > 0:

    print(
        "Accuracy when Team 1 won toss:",
        round(
            accuracy_score(
                team1_won_toss["actual"],
                team1_won_toss["predicted"]
            ) * 100,
            2
        ),
        "%"
    )

if len(team2_won_toss) > 0:

    print(
        "Accuracy when Team 2 won toss:",
        round(
            accuracy_score(
                team2_won_toss["actual"],
                team2_won_toss["predicted"]
            ) * 100,
            2
        ),
        "%"
    )


# -------------------------------------------------
# Venue Difference Analysis
# -------------------------------------------------

results["venue_difference"] = abs(
    results["team1_venue_win_rate"]
    -
    results["team2_venue_win_rate"]
)

similar_venue_strength = results[
    results["venue_difference"] <= 0.10
]

if len(similar_venue_strength) > 0:

    similar_venue_accuracy = accuracy_score(
        similar_venue_strength["actual"],
        similar_venue_strength["predicted"]
    )

    print(
        "\n--- SIMILAR VENUE PERFORMANCE ---"
    )

    print(
        "Matches:",
        len(similar_venue_strength)
    )

    print(
        "Accuracy:",
        round(
            similar_venue_accuracy * 100,
            2
        ),
        "%"
    )


# -------------------------------------------------
# Confidence Analysis
# -------------------------------------------------

results["confidence"] = results[
    [
        "team1_probability",
        "team2_probability"
    ]
].max(axis=1)

high_confidence = results[
    results["confidence"] >= 0.70
]

medium_confidence = results[
    (results["confidence"] >= 0.60)
    &
    (results["confidence"] < 0.70)
]

low_confidence = results[
    results["confidence"] < 0.60
]


# -------------------------------------------------
# High Confidence
# -------------------------------------------------

if len(high_confidence) > 0:

    high_conf_accuracy = accuracy_score(
        high_confidence["actual"],
        high_confidence["predicted"]
    )

    print(
        "\n--- HIGH CONFIDENCE PREDICTIONS ---"
    )

    print(
        "Matches:",
        len(high_confidence)
    )

    print(
        "Accuracy:",
        round(
            high_conf_accuracy * 100,
            2
        ),
        "%"
    )


# -------------------------------------------------
# Medium Confidence
# -------------------------------------------------

if len(medium_confidence) > 0:

    medium_conf_accuracy = accuracy_score(
        medium_confidence["actual"],
        medium_confidence["predicted"]
    )

    print(
        "\n--- MEDIUM CONFIDENCE PREDICTIONS ---"
    )

    print(
        "Matches:",
        len(medium_confidence)
    )

    print(
        "Accuracy:",
        round(
            medium_conf_accuracy * 100,
            2
        ),
        "%"
    )


# -------------------------------------------------
# Low Confidence
# -------------------------------------------------

if len(low_confidence) > 0:

    low_conf_accuracy = accuracy_score(
        low_confidence["actual"],
        low_confidence["predicted"]
    )

    print(
        "\n--- LOW CONFIDENCE PREDICTIONS ---"
    )

    print(
        "Matches:",
        len(low_confidence)
    )

    print(
        "Accuracy:",
        round(
            low_conf_accuracy * 100,
            2
        ),
        "%"
    )


# -------------------------------------------------
# Most Confident Wrong Predictions
# -------------------------------------------------

errors = results[
    results["correct_prediction"] == False
].copy()

errors = errors.sort_values(
    "confidence",
    ascending=False
)

print(
    "\n--- MOST CONFIDENT WRONG PREDICTIONS ---"
)

print(
    errors[
        [
            "date",
            "team1",
            "team2",
            "venue",
            "winner",
            "team1_probability",
            "team2_probability",
            "confidence"
        ]
    ].head(10).to_string(
        index=False
    )
)


# -------------------------------------------------
# Save Analysis
# -------------------------------------------------

output_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\prediction_error_analysis.csv"
)

results.to_csv(
    output_file,
    index=False
)

print(
    "\nError analysis dataset created:"
)

print(
    output_file
)