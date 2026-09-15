import pandas as pd
from pathlib import Path


ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

LOG_FILE = (
    ROOT
    / "data"
    / "shadow_v12_log.csv"
)


print("========================================")
print("CANDIDATE v1.2 - SHADOW MONITORING")
print("========================================")


# -------------------------------------------------
# Check file
# -------------------------------------------------

if not LOG_FILE.exists():

    print("\nShadow log does not exist.")

    print(
        "Expected file:",
        LOG_FILE
    )

    raise SystemExit(1)


# -------------------------------------------------
# Load shadow log
# -------------------------------------------------

df = pd.read_csv(
    LOG_FILE
)


if len(df) == 0:

    print("\nShadow log is empty.")

    raise SystemExit(0)


# -------------------------------------------------
# Normalize important columns
# -------------------------------------------------

df["result_status"] = (
    df["result_status"]
    .fillna("PENDING")
    .astype(str)
    .str.upper()
)


# Convert correctness carefully
def normalize_correct(value):

    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text == "true":
        return True

    if text == "false":
        return False

    return None


df["correct_normalized"] = (
    df["correct_prediction"]
    .apply(normalize_correct)
)


# -------------------------------------------------
# Overall counts
# -------------------------------------------------

total_predictions = len(
    df
)


pending = df[
    df["result_status"]
    ==
    "PENDING"
]


completed = df[
    df["result_status"]
    ==
    "COMPLETED"
]


completed_count = len(
    completed
)


pending_count = len(
    pending
)


correct_count = (
    completed[
        completed[
            "correct_normalized"
        ]
        ==
        True
    ]
    .shape[0]
)


incorrect_count = (
    completed[
        completed[
            "correct_normalized"
        ]
        ==
        False
    ]
    .shape[0]
)


# -------------------------------------------------
# Accuracy
# -------------------------------------------------

if completed_count > 0:

    accuracy = (
        correct_count
        /
        completed_count
    )

else:

    accuracy = None


# -------------------------------------------------
# Main summary
# -------------------------------------------------

print("\n========================================")
print("SHADOW SUMMARY")
print("========================================")

print(
    "Model:",
    "Candidate v1.2"
)

print(
    "Total predictions:",
    total_predictions
)

print(
    "Pending:",
    pending_count
)

print(
    "Completed:",
    completed_count
)

print(
    "Correct:",
    correct_count
)

print(
    "Incorrect:",
    incorrect_count
)


if accuracy is not None:

    print(
        "Live shadow accuracy:",
        f"{accuracy * 100:.2f}%"
    )

else:

    print(
        "Live shadow accuracy:",
        "N/A"
    )


# -------------------------------------------------
# Completed prediction details
# -------------------------------------------------

print("\n========================================")
print("COMPLETED PREDICTIONS")
print("========================================")


if completed_count == 0:

    print(
        "No completed shadow predictions."
    )

else:

    display_columns = [
        "prediction_timestamp",
        "team1",
        "team2",
        "venue",
        "predicted_winner",
        "actual_winner",
        "correct_prediction"
    ]


    print(
        completed[
            display_columns
        ].to_string(
            index=False
        )
    )


# -------------------------------------------------
# Pending prediction details
# -------------------------------------------------

print("\n========================================")
print("PENDING PREDICTIONS")
print("========================================")


if pending_count == 0:

    print(
        "No pending shadow predictions."
    )

else:

    display_columns = [
        "prediction_timestamp",
        "team1",
        "team2",
        "venue",
        "predicted_winner"
    ]


    print(
        pending[
            display_columns
        ].to_string(
            index=False
        )
    )


# -------------------------------------------------
# Confidence summary
# -------------------------------------------------

if (
    "team1_probability"
    in df.columns
    and
    "team2_probability"
    in df.columns
):

    df[
        "prediction_confidence"
    ] = df[
        [
            "team1_probability",
            "team2_probability"
        ]
    ].max(
        axis=1
    )


    average_confidence = (
        df[
            "prediction_confidence"
        ]
        .mean()
    )


    print("\n========================================")
    print("CONFIDENCE")
    print("========================================")

    print(
        "Average prediction confidence:",
        f"{average_confidence * 100:.2f}%"
    )


# -------------------------------------------------
# Warning for small sample
# -------------------------------------------------

print("\n========================================")
print("INTERPRETATION")
print("========================================")


if completed_count < 5:

    print(
        "WARNING:"
    )

    print(
        "Too few completed predictions "
        "to judge Candidate v1.2 reliably."
    )

    print(
        "Continue shadow testing before "
        "considering model promotion."
    )

elif completed_count < 20:

    print(
        "Candidate v1.2 has some "
        "forward-test evidence,"
    )

    print(
        "but the sample is still small."
    )

    print(
        "Continue shadow monitoring."
    )

else:

    print(
        "Candidate v1.2 now has a more "
        "meaningful forward-test sample."
    )

    print(
        "Compare its results against "
        "Champion v1.0 before promotion."
    )


print("\nChampion v1.0 remains production.")