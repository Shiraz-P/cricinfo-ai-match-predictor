import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

CHAMPION_MODEL_FILE = (
    PROJECT_ROOT / "model" / "cricket_model.pkl"
)

CHALLENGER_MODEL_FILE = (
    PROJECT_ROOT / "model" / "cricket_model_challenger_v11.pkl"
)

CHAMPION_DATA_FILE = (
    PROJECT_ROOT / "data" / "matches_features.csv"
)

CHALLENGER_DATA_FILE = (
    PROJECT_ROOT / "data" / "matches_candidate_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "champion_challenger_v11_comparison.csv"
)

print("========================================")
print("CHAMPION vs CHALLENGER")
print("========================================")


# -------------------------------------------------
# Load model packages
# -------------------------------------------------

champion_package = joblib.load(
    CHAMPION_MODEL_FILE
)

challenger_package = joblib.load(
    CHALLENGER_MODEL_FILE
)


# -------------------------------------------------
# Extract Champion model
# -------------------------------------------------

if isinstance(champion_package, dict):

    champion_model = champion_package.get(
        "model"
    )

    if champion_model is None:

        raise ValueError(
            "Champion package does not contain model."
        )

else:

    champion_model = champion_package


# -------------------------------------------------
# Extract Challenger model + features
# -------------------------------------------------

if isinstance(challenger_package, dict):

    challenger_model = challenger_package.get(
        "model"
    )

    challenger_features = challenger_package.get(
        "features"
    )

    if challenger_model is None:

        raise ValueError(
            "Challenger package does not contain model."
        )

    if challenger_features is None:

        raise ValueError(
            "Challenger package does not contain features."
        )

else:

    challenger_model = challenger_package

    challenger_features = [
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


# -------------------------------------------------
# Champion feature list
# -------------------------------------------------

champion_features = [
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


print(
    "\nChampion features:",
    len(champion_features)
)

print(
    "Challenger features:",
    len(challenger_features)
)


# -------------------------------------------------
# Load datasets
# -------------------------------------------------

champion_df = pd.read_csv(
    CHAMPION_DATA_FILE
)

challenger_df = pd.read_csv(
    CHALLENGER_DATA_FILE
)


for df in [
    champion_df,
    challenger_df
]:

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


# -------------------------------------------------
# Sort chronologically
# -------------------------------------------------

champion_df = champion_df.sort_values(
    "date",
    kind="stable"
).reset_index(drop=True)

challenger_df = challenger_df.sort_values(
    "date",
    kind="stable"
).reset_index(drop=True)


# -------------------------------------------------
# Champion chronological 80/20 test set
# -------------------------------------------------

champion_split = int(
    len(champion_df) * 0.80
)

champion_test = champion_df.iloc[
    champion_split:
].copy()


print("\n========================================")
print("COMMON TEST SET")
print("========================================")

print(
    "Champion total rows:",
    len(champion_df)
)

print(
    "Champion test matches:",
    len(champion_test)
)

print(
    "Test period:",
    champion_test["date"].min(),
    "to",
    champion_test["date"].max()
)


# -------------------------------------------------
# Exact match key
#
# IMPORTANT:
#
# We intentionally keep team1/team2 ORDER.
#
# We also include winner because this comparison
# needs to identify the exact historical record.
#
# Occurrence handles the unlikely case where two
# completely identical records exist.
# -------------------------------------------------

def prepare_exact_key(df):

    result = df.copy()

    result["_base_key"] = (
        result["date"]
        .dt.strftime("%Y-%m-%d")
        + "|"
        + result["team1"].astype(str)
        + "|"
        + result["team2"].astype(str)
        + "|"
        + result["winner"].astype(str)
    )

    result["_occurrence"] = (
        result.groupby(
            "_base_key"
        ).cumcount()
    )

    result["_comparison_key"] = (
        result["_base_key"]
        + "|"
        + result["_occurrence"].astype(str)
    )

    return result


champion_test = prepare_exact_key(
    champion_test
)

challenger_prepared = prepare_exact_key(
    challenger_df
)


# -------------------------------------------------
# Match Champion test records against Challenger
# -------------------------------------------------

champion_keys = set(
    champion_test["_comparison_key"]
)

challenger_test = challenger_prepared[
    challenger_prepared[
        "_comparison_key"
    ].isin(
        champion_keys
    )
].copy()


challenger_keys = set(
    challenger_test[
        "_comparison_key"
    ]
)


missing = champion_test[
    ~champion_test[
        "_comparison_key"
    ].isin(
        challenger_keys
    )
].copy()


print(
    "\nMissing Champion matches in Challenger:",
    len(missing)
)


if len(missing) > 0:

    print(
        "\nFirst missing records:"
    )

    print(
        missing[
            [
                "date",
                "team1",
                "team2",
                "winner"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# -------------------------------------------------
# Keep common records only
# -------------------------------------------------

champion_common = champion_test[
    champion_test[
        "_comparison_key"
    ].isin(
        challenger_keys
    )
].copy()


champion_common = (
    champion_common
    .sort_values(
        "_comparison_key"
    )
    .reset_index(drop=True)
)


challenger_test = (
    challenger_test
    .sort_values(
        "_comparison_key"
    )
    .reset_index(drop=True)
)


print(
    "Successfully matched:",
    len(champion_common)
)


# -------------------------------------------------
# Basic safety validations
# -------------------------------------------------

if len(champion_common) == 0:

    raise ValueError(
        "No common matches found."
    )


if len(champion_common) != len(
    challenger_test
):

    raise ValueError(
        "Champion and Challenger row counts differ."
    )


key_alignment = (
    champion_common[
        "_comparison_key"
    ].values
    ==
    challenger_test[
        "_comparison_key"
    ].values
).all()


print(
    "Key alignment:",
    key_alignment
)


if not key_alignment:

    raise ValueError(
        "Comparison keys are not aligned."
    )


# -------------------------------------------------
# Validate date
# -------------------------------------------------

date_alignment = (
    champion_common[
        "date"
    ].values
    ==
    challenger_test[
        "date"
    ].values
).all()


print(
    "Date alignment:",
    date_alignment
)


# -------------------------------------------------
# Validate team1
# -------------------------------------------------

team1_alignment = (
    champion_common[
        "team1"
    ].astype(str).values
    ==
    challenger_test[
        "team1"
    ].astype(str).values
).all()


print(
    "Team1 alignment:",
    team1_alignment
)


# -------------------------------------------------
# Validate team2
# -------------------------------------------------

team2_alignment = (
    champion_common[
        "team2"
    ].astype(str).values
    ==
    challenger_test[
        "team2"
    ].astype(str).values
).all()


print(
    "Team2 alignment:",
    team2_alignment
)


# -------------------------------------------------
# Validate winner
# -------------------------------------------------

winner_alignment = (
    champion_common[
        "winner"
    ].astype(str).values
    ==
    challenger_test[
        "winner"
    ].astype(str).values
).all()


print(
    "Winner alignment:",
    winner_alignment
)


if not (
    date_alignment
    and
    team1_alignment
    and
    team2_alignment
    and
    winner_alignment
):

    raise ValueError(
        "Historical records are not aligned safely."
    )


# -------------------------------------------------
# Actual outcome
#
# Both datasets now have identical team ordering,
# therefore team1_won can be directly compared.
# -------------------------------------------------

y_actual = (
    champion_common[
        "winner"
    ]
    ==
    champion_common[
        "team1"
    ]
).astype(int)


challenger_actual = (
    challenger_test[
        "winner"
    ]
    ==
    challenger_test[
        "team1"
    ]
).astype(int)


target_alignment = (
    y_actual.values
    ==
    challenger_actual.values
).all()


print(
    "Target alignment:",
    target_alignment
)


if not target_alignment:

    raise ValueError(
        "team1_won targets are not aligned."
    )


# -------------------------------------------------
# Champion predictions
# -------------------------------------------------

X_champion = champion_common[
    champion_features
]


champion_predictions = (
    champion_model.predict(
        X_champion
    )
)


# -------------------------------------------------
# Challenger predictions
# -------------------------------------------------

X_challenger = challenger_test[
    challenger_features
]


challenger_predictions = (
    challenger_model.predict(
        X_challenger
    )
)


# -------------------------------------------------
# Accuracy
# -------------------------------------------------

champion_accuracy = accuracy_score(
    y_actual,
    champion_predictions
)


challenger_accuracy = accuracy_score(
    y_actual,
    challenger_predictions
)


print("\n========================================")
print("HEAD-TO-HEAD RESULTS")
print("========================================")


print(
    "Compared matches:",
    len(y_actual)
)


print(
    "Champion v1.0 accuracy:",
    round(
        champion_accuracy,
        4
    )
)


print(
    "Challenger v1.1 accuracy:",
    round(
        challenger_accuracy,
        4
    )
)


difference = (
    challenger_accuracy
    -
    champion_accuracy
)


print(
    "Accuracy difference:",
    round(
        difference,
        4
    )
)


print(
    "Percentage-point difference:",
    round(
        difference * 100,
        2
    )
)


# -------------------------------------------------
# Detailed comparison
# -------------------------------------------------

comparison = pd.DataFrame({

    "date":
        champion_common[
            "date"
        ].values,

    "team1":
        champion_common[
            "team1"
        ].values,

    "team2":
        champion_common[
            "team2"
        ].values,

    "winner":
        champion_common[
            "winner"
        ].values,

    "actual_team1_won":
        y_actual.values,

    "champion_prediction":
        champion_predictions,

    "challenger_prediction":
        challenger_predictions
})


comparison[
    "champion_correct"
] = (
    comparison[
        "actual_team1_won"
    ]
    ==
    comparison[
        "champion_prediction"
    ]
)


comparison[
    "challenger_correct"
] = (
    comparison[
        "actual_team1_won"
    ]
    ==
    comparison[
        "challenger_prediction"
    ]
)


comparison[
    "models_disagree"
] = (
    comparison[
        "champion_prediction"
    ]
    !=
    comparison[
        "challenger_prediction"
    ]
)


# -------------------------------------------------
# Disagreement analysis
# -------------------------------------------------

disagreements = comparison[
    comparison[
        "models_disagree"
    ]
].copy()


champion_only_correct = disagreements[
    disagreements[
        "champion_correct"
    ]
    &
    ~disagreements[
        "challenger_correct"
    ]
]


challenger_only_correct = disagreements[
    disagreements[
        "challenger_correct"
    ]
    &
    ~disagreements[
        "champion_correct"
    ]
]


same_predictions = comparison[
    ~comparison[
        "models_disagree"
    ]
]


both_correct = same_predictions[
    same_predictions[
        "champion_correct"
    ]
]


both_wrong = same_predictions[
    ~same_predictions[
        "champion_correct"
    ]
]


print("\n========================================")
print("DISAGREEMENT ANALYSIS")
print("========================================")


print(
    "Matches where models disagree:",
    len(disagreements)
)


print(
    "Champion correct / Challenger wrong:",
    len(champion_only_correct)
)


print(
    "Challenger correct / Champion wrong:",
    len(challenger_only_correct)
)


print(
    "Both models correct:",
    len(both_correct)
)


print(
    "Both models wrong:",
    len(both_wrong)
)


# -------------------------------------------------
# Decision
# -------------------------------------------------

print("\n========================================")
print("MODEL DECISION")
print("========================================")


if challenger_accuracy > champion_accuracy:

    print(
        "RESULT: Challenger v1.1 performed better."
    )

    print(
        "STATUS: Challenger is a promotion candidate."
    )

    print(
        "Do NOT promote yet. "
        "Final validation is still required."
    )


elif challenger_accuracy < champion_accuracy:

    print(
        "RESULT: Champion v1.0 performed better."
    )

    print(
        "STATUS: Keep Champion v1.0."
    )


else:

    print(
        "RESULT: Champion and Challenger tied."
    )

    print(
        "STATUS: Keep Champion v1.0."
    )


# -------------------------------------------------
# Save comparison
# -------------------------------------------------

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n========================================")
print("COMPARISON SAVED")
print("========================================")


print(
    "File:",
    OUTPUT_FILE
)


print(
    "\nChampion v1.0 was NOT modified."
)

print(
    "Challenger v1.1 was NOT modified."
)