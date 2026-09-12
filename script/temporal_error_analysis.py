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
# Chronological 80/20 Split
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
# Prediction
# -------------------------------------------------

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)

team1_probabilities = probabilities[:, 1]
team2_probabilities = probabilities[:, 0]


# -------------------------------------------------
# Build Analysis Dataset
# -------------------------------------------------

results = df.iloc[
    split_index:
].copy()

results["actual"] = y_test.values

results["predicted"] = predictions

results[
    "team1_probability"
] = team1_probabilities

results[
    "team2_probability"
] = team2_probabilities

results[
    "correct_prediction"
] = (
    results["actual"]
    ==
    results["predicted"]
)


# -------------------------------------------------
# Confidence
# -------------------------------------------------

results["confidence"] = results[
    [
        "team1_probability",
        "team2_probability"
    ]
].max(
    axis=1
)


def confidence_level(value):

    if value >= 0.70:
        return "HIGH"

    elif value >= 0.60:
        return "MEDIUM"

    else:
        return "LOW"


results[
    "confidence_level"
] = results[
    "confidence"
].apply(
    confidence_level
)


# -------------------------------------------------
# Overall Performance
# -------------------------------------------------

overall_accuracy = accuracy_score(
    results["actual"],
    results["predicted"]
)


print("\n================================")
print("TEMPORAL ERROR ANALYSIS")
print("================================")

print(
    "Test matches:",
    len(results)
)

print(
    "Overall Accuracy:",
    round(
        overall_accuracy * 100,
        2
    ),
    "%"
)


# -------------------------------------------------
# Performance by Year
# -------------------------------------------------

results["year"] = results[
    "date"
].dt.year


print(
    "\n--- ACCURACY BY YEAR ---"
)

year_results = (
    results
    .groupby(
        "year"
    )
    .agg(
        matches=(
            "correct_prediction",
            "size"
        ),

        correct=(
            "correct_prediction",
            "sum"
        )
    )
)

year_results[
    "accuracy"
] = (
    year_results[
        "correct"
    ]
    /
    year_results[
        "matches"
    ]
    *
    100
)

print(
    year_results.round(
        2
    ).to_string()
)


# -------------------------------------------------
# Accuracy by Confidence Level
# -------------------------------------------------

print(
    "\n--- ACCURACY BY CONFIDENCE ---"
)

confidence_results = (
    results
    .groupby(
        "confidence_level"
    )
    .agg(
        matches=(
            "correct_prediction",
            "size"
        ),

        correct=(
            "correct_prediction",
            "sum"
        ),

        average_confidence=(
            "confidence",
            "mean"
        )
    )
)

confidence_results[
    "accuracy"
] = (
    confidence_results[
        "correct"
    ]
    /
    confidence_results[
        "matches"
    ]
    *
    100
)

confidence_results[
    "average_confidence"
] = (
    confidence_results[
        "average_confidence"
    ]
    *
    100
)

print(
    confidence_results.round(
        2
    ).to_string()
)


# -------------------------------------------------
# Similar Strength Teams
# -------------------------------------------------

results[
    "historical_difference"
] = abs(
    results[
        "team1_historical_win_rate"
    ]
    -
    results[
        "team2_historical_win_rate"
    ]
)


similar_strength = results[
    results[
        "historical_difference"
    ]
    <= 0.10
]


print(
    "\n--- SIMILAR STRENGTH MATCHES ---"
)

print(
    "Matches:",
    len(similar_strength)
)

if len(similar_strength) > 0:

    similar_accuracy = accuracy_score(
        similar_strength["actual"],
        similar_strength["predicted"]
    )

    print(
        "Accuracy:",
        round(
            similar_accuracy * 100,
            2
        ),
        "%"
    )


# -------------------------------------------------
# Performance by Toss Winner
# -------------------------------------------------

print(
    "\n--- TOSS IMPACT ---"
)

team1_toss = results[
    results[
        "toss_winner_is_team1"
    ]
    ==
    1
]

team2_toss = results[
    results[
        "toss_winner_is_team1"
    ]
    ==
    0
]


if len(team1_toss) > 0:

    print(
        "Accuracy when Team 1 won toss:",
        round(
            accuracy_score(
                team1_toss["actual"],
                team1_toss["predicted"]
            )
            *
            100,
            2
        ),
        "%"
    )


if len(team2_toss) > 0:

    print(
        "Accuracy when Team 2 won toss:",
        round(
            accuracy_score(
                team2_toss["actual"],
                team2_toss["predicted"]
            )
            *
            100,
            2
        ),
        "%"
    )


# -------------------------------------------------
# Most Difficult Teams
# -------------------------------------------------

team_records = []

all_test_teams = sorted(
    set(
        results["team1"]
    ).union(
        set(
            results["team2"]
        )
    )
)


for team in all_test_teams:

    team_matches = results[
        (results["team1"] == team)
        |
        (results["team2"] == team)
    ]

    if len(team_matches) < 10:
        continue

    team_accuracy = (
        team_matches[
            "correct_prediction"
        ].mean()
        *
        100
    )

    team_records.append({
        "team": team,
        "matches": len(
            team_matches
        ),
        "accuracy": team_accuracy
    })


team_analysis = pd.DataFrame(
    team_records
)


if len(team_analysis) > 0:

    team_analysis = (
        team_analysis
        .sort_values(
            by="accuracy",
            ascending=True
        )
    )


    print(
        "\n--- HARDEST TEAMS TO PREDICT ---"
    )

    print(
        team_analysis
        .head(15)
        .round(2)
        .to_string(
            index=False
        )
    )


# -------------------------------------------------
# Most Confident Wrong Predictions
# -------------------------------------------------

wrong = results[
    results[
        "correct_prediction"
    ]
    ==
    False
].copy()


wrong = wrong.sort_values(
    by="confidence",
    ascending=False
)


print(
    "\n--- MOST CONFIDENT WRONG PREDICTIONS ---"
)

print(
    wrong[
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
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# -------------------------------------------------
# Save Analysis
# -------------------------------------------------

output_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\temporal_error_analysis.csv"
)

results.to_csv(
    output_file,
    index=False
)


print(
    "\nTemporal error analysis saved to:"
)

print(
    output_file
)