import pandas as pd


# -------------------------------------------------
# Prediction log file
# -------------------------------------------------

log_file = r"K:\Python\Cricinfo_AI_Project\data\prediction_log.csv"

df = pd.read_csv(log_file)


# -------------------------------------------------
# Ensure monitoring columns exist
# -------------------------------------------------

if "actual_winner" not in df.columns:
    df["actual_winner"] = pd.NA

if "correct_prediction" not in df.columns:
    df["correct_prediction"] = pd.NA

if "result_status" not in df.columns:
    df["result_status"] = "PENDING"


# -------------------------------------------------
# Normalize result status
# -------------------------------------------------

df["result_status"] = (
    df["result_status"]
    .astype("string")
    .str.strip()
    .str.upper()
)


# -------------------------------------------------
# Normalize correct_prediction
# -------------------------------------------------

df["correct_prediction"] = (
    df["correct_prediction"]
    .astype("string")
    .str.strip()
    .str.lower()
)


# -------------------------------------------------
# Basic counts
# -------------------------------------------------

total_predictions = len(df)

pending = df[
    df["result_status"] == "PENDING"
]

completed = df[
    df["result_status"] == "COMPLETED"
]

correct = completed[
    completed["correct_prediction"] == "true"
]

incorrect = completed[
    completed["correct_prediction"] == "false"
]


# -------------------------------------------------
# Print monitoring summary
# -------------------------------------------------

print("\n================================")
print("MODEL MONITORING SUMMARY")
print("================================")

print(
    "Total predictions:",
    total_predictions
)

print(
    "Pending predictions:",
    len(pending)
)

print(
    "Completed predictions:",
    len(completed)
)

print(
    "Correct predictions:",
    len(correct)
)

print(
    "Incorrect predictions:",
    len(incorrect)
)


# -------------------------------------------------
# Live accuracy
# -------------------------------------------------

if len(completed) > 0:

    live_accuracy = (
        len(correct)
        /
        len(completed)
    )

    print(
        "Live Accuracy:",
        round(
            live_accuracy * 100,
            2
        ),
        "%"
    )

else:

    print(
        "Live Accuracy: "
        "Not available yet"
    )


# -------------------------------------------------
# Accuracy by confidence
# -------------------------------------------------

print("\n--- ACCURACY BY CONFIDENCE ---")

for confidence_level in [
    "HIGH",
    "MEDIUM",
    "LOW"
]:

    confidence_data = completed[
        completed["confidence"]
        == confidence_level
    ]

    if len(confidence_data) > 0:

        confidence_correct = confidence_data[
            confidence_data["correct_prediction"]
            == "true"
        ]

        confidence_accuracy = (
            len(confidence_correct)
            /
            len(confidence_data)
        )

        print(
            confidence_level,
            ":",
            round(
                confidence_accuracy * 100,
                2
            ),
            "%",
            "- Matches:",
            len(confidence_data)
        )

    else:

        print(
            confidence_level,
            ": No completed predictions"
        )


# -------------------------------------------------
# Accuracy by model version
# -------------------------------------------------

print("\n--- ACCURACY BY MODEL VERSION ---")

if "model_version" in df.columns:

    versions = df[
        "model_version"
    ].dropna().unique()

    for version in versions:

        version_data = completed[
            completed["model_version"]
            == version
        ]

        if len(version_data) > 0:

            version_correct = version_data[
                version_data["correct_prediction"]
                == "true"
            ]

            version_accuracy = (
                len(version_correct)
                /
                len(version_data)
            )

            print(
                "Version",
                version,
                ":",
                round(
                    version_accuracy * 100,
                    2
                ),
                "%",
                "- Matches:",
                len(version_data)
            )

        else:

            print(
                "Version",
                version,
                ": No completed predictions"
            )

else:

    print(
        "model_version column not available"
    )


# -------------------------------------------------
# Recent predictions
# -------------------------------------------------

print("\n--- RECENT PREDICTIONS ---")

recent_columns = [
    "prediction_time",
    "team1",
    "team2",
    "predicted_winner",
    "actual_winner",
    "confidence",
    "correct_prediction",
    "result_status"
]

recent_columns = [
    column
    for column in recent_columns
    if column in df.columns
]

print(
    df[
        recent_columns
    ]
    .tail(10)
    .to_string(
        index=False
    )
)