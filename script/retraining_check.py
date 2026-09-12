import pandas as pd


# -------------------------------------------------
# Prediction log file
# -------------------------------------------------

log_file = r"K:\Python\Cricinfo_AI_Project\data\prediction_log.csv"

df = pd.read_csv(log_file)


# -------------------------------------------------
# Configuration / Thresholds
# -------------------------------------------------

minimum_completed_predictions = 30

baseline_accuracy = 71.82

warning_accuracy = 65.00


# -------------------------------------------------
# Ensure monitoring columns exist
# -------------------------------------------------

if "result_status" not in df.columns:
    df["result_status"] = "PENDING"


# -------------------------------------------------
# Select completed predictions only
# -------------------------------------------------

completed = df[
    df["result_status"].isin(
        ["CORRECT", "INCORRECT"]
    )
].copy()


correct = completed[
    completed["result_status"] == "CORRECT"
]


completed_count = len(completed)


# -------------------------------------------------
# Calculate live accuracy
# -------------------------------------------------

if completed_count > 0:

    live_accuracy = (
        len(correct)
        /
        completed_count
        *
        100
    )

else:

    live_accuracy = None


# -------------------------------------------------
# Report
# -------------------------------------------------

print("\n================================")
print("RETRAINING READINESS CHECK")
print("================================")

print(
    "Baseline holdout accuracy:",
    baseline_accuracy,
    "%"
)

print(
    "Minimum completed predictions required:",
    minimum_completed_predictions
)

print(
    "Current completed predictions:",
    completed_count
)


# -------------------------------------------------
# Not enough real results yet
# -------------------------------------------------

if completed_count < minimum_completed_predictions:

    remaining = (
        minimum_completed_predictions
        -
        completed_count
    )

    print("\nSTATUS: NOT READY FOR RETRAINING DECISION")

    print(
        "More completed predictions needed:",
        remaining
    )

    if live_accuracy is not None:

        print(
            "Current live accuracy:",
            round(
                live_accuracy,
                2
            ),
            "%"
        )

    else:

        print(
            "Current live accuracy: "
            "Not available yet"
        )

    print(
        "\nRecommendation:"
    )

    print(
        "Continue collecting predictions "
        "and actual match results."
    )


# -------------------------------------------------
# Enough results available
# -------------------------------------------------

else:

    print(
        "\nCurrent live accuracy:",
        round(
            live_accuracy,
            2
        ),
        "%"
    )

    difference = (
        live_accuracy
        -
        baseline_accuracy
    )

    print(
        "Difference from baseline:",
        round(
            difference,
            2
        ),
        "percentage points"
    )


    # ---------------------------------------------
    # Strong degradation
    # ---------------------------------------------

    if live_accuracy < warning_accuracy:

        print(
            "\nSTATUS: INVESTIGATE / CONSIDER RETRAINING"
        )

        print(
            "Live accuracy has fallen below",
            warning_accuracy,
            "%"
        )

        print(
            "\nRecommended actions:"
        )

        print(
            "1. Review incorrect predictions."
        )

        print(
            "2. Check for data or concept drift."
        )

        print(
            "3. Review feature quality."
        )

        print(
            "4. Add recent completed matches "
            "to the training dataset."
        )

        print(
            "5. Train a candidate new model."
        )

        print(
            "6. Compare the new model against "
            "version 1.0 before replacing it."
        )


    # ---------------------------------------------
    # Performance acceptable
    # ---------------------------------------------

    else:

        print(
            "\nSTATUS: MODEL PERFORMANCE ACCEPTABLE"
        )

        print(
            "No retraining required based "
            "on the current rule."
        )

        print(
            "Continue monitoring."
        )