import pandas as pd


# -------------------------------------------------
# Shadow comparison log
# -------------------------------------------------

log_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\shadow_comparison_log.csv"
)


# -------------------------------------------------
# Load data
# -------------------------------------------------

df = pd.read_csv(
    log_file
)


# -------------------------------------------------
# Normalize important columns
# -------------------------------------------------

if "result_status" not in df.columns:
    df["result_status"] = "PENDING"

df["result_status"] = (
    df["result_status"]
    .fillna("PENDING")
    .astype(str)
)


for column in [
    "champion_correct",
    "challenger_correct",
    "actual_winner"
]:

    if column not in df.columns:
        df[column] = ""


# -------------------------------------------------
# Header
# -------------------------------------------------

print("\n================================")
print("SHADOW MODEL MONITORING")
print("================================")


# -------------------------------------------------
# Basic counts
# -------------------------------------------------

total_predictions = len(df)

completed = df[
    df["result_status"] == "COMPLETED"
].copy()

pending = df[
    df["result_status"] == "PENDING"
].copy()


print(
    "Total shadow predictions:",
    total_predictions
)

print(
    "Completed predictions:",
    len(completed)
)

print(
    "Pending predictions:",
    len(pending)
)


# -------------------------------------------------
# Agreement / Disagreement
# -------------------------------------------------

agreement_mask = (
    df["models_agree"]
    .astype(str)
    .str.lower()
    .eq("true")
)

agreement_count = (
    agreement_mask.sum()
)

disagreement_count = (
    total_predictions
    -
    agreement_count
)


if total_predictions > 0:

    agreement_rate = (
        agreement_count
        /
        total_predictions
        *
        100
    )

    disagreement_rate = (
        disagreement_count
        /
        total_predictions
        *
        100
    )

else:

    agreement_rate = 0
    disagreement_rate = 0


print(
    "\n--- MODEL AGREEMENT ---"
)

print(
    "Agreement count:",
    agreement_count
)

print(
    "Disagreement count:",
    disagreement_count
)

print(
    "Agreement rate:",
    round(
        agreement_rate,
        2
    ),
    "%"
)

print(
    "Disagreement rate:",
    round(
        disagreement_rate,
        2
    ),
    "%"
)


# -------------------------------------------------
# Probability Difference
# -------------------------------------------------

print(
    "\n--- PROBABILITY DIFFERENCE ---"
)


if (
    "probability_difference_pp"
    in df.columns
    and
    len(df) > 0
):

    probability_values = (
        pd.to_numeric(
            df[
                "probability_difference_pp"
            ],
            errors="coerce"
        )
        .dropna()
    )


    if len(probability_values) > 0:

        print(
            "Average probability difference:",
            round(
                probability_values.mean(),
                2
            ),
            "percentage points"
        )

        print(
            "Maximum probability difference:",
            round(
                probability_values.max(),
                2
            ),
            "percentage points"
        )

        print(
            "Minimum probability difference:",
            round(
                probability_values.min(),
                2
            ),
            "percentage points"
        )

    else:

        print(
            "No probability differences available."
        )

else:

    print(
        "Probability difference data "
        "not available."
    )


# -------------------------------------------------
# Live Accuracy
# -------------------------------------------------

print(
    "\n--- LIVE SHADOW ACCURACY ---"
)


if len(completed) > 0:

    champion_correct = (
        completed[
            "champion_correct"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    challenger_correct = (
        completed[
            "challenger_correct"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
    )


    champion_accuracy = (
        champion_correct.mean()
        *
        100
    )

    challenger_accuracy = (
        challenger_correct.mean()
        *
        100
    )


    print(
        "Champion Accuracy:",
        round(
            champion_accuracy,
            2
        ),
        "%"
    )

    print(
        "Challenger Accuracy:",
        round(
            challenger_accuracy,
            2
        ),
        "%"
    )


    if (
        challenger_accuracy
        >
        champion_accuracy
    ):

        print(
            "Current shadow leader:"
            " CHALLENGER"
        )

    elif (
        champion_accuracy
        >
        challenger_accuracy
    ):

        print(
            "Current shadow leader:"
            " CHAMPION"
        )

    else:

        print(
            "Current shadow result:"
            " TIED"
        )


else:

    print(
        "Live shadow accuracy:"
        " Not available yet."
    )

    print(
        "Ground truth is still pending."
    )


# -------------------------------------------------
# Confidence Distribution
# -------------------------------------------------

print(
    "\n--- CHAMPION CONFIDENCE DISTRIBUTION ---"
)

if (
    "champion_confidence_level"
    in df.columns
):

    print(
        df[
            "champion_confidence_level"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

else:

    print(
        "Champion confidence data unavailable."
    )


print(
    "\n--- CHALLENGER CONFIDENCE DISTRIBUTION ---"
)

if (
    "challenger_confidence_level"
    in df.columns
):

    print(
        df[
            "challenger_confidence_level"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

else:

    print(
        "Challenger confidence data unavailable."
    )


# -------------------------------------------------
# Completed Accuracy by Confidence
# -------------------------------------------------

if len(completed) > 0:

    print(
        "\n--- CHAMPION ACCURACY BY CONFIDENCE ---"
    )

    for level in [
        "HIGH",
        "MEDIUM",
        "LOW"
    ]:

        subset = completed[
            completed[
                "champion_confidence_level"
            ]
            ==
            level
        ]

        if len(subset) > 0:

            accuracy = (
                subset[
                    "champion_correct"
                ]
                .astype(str)
                .str.lower()
                .eq("true")
                .mean()
                *
                100
            )

            print(
                level,
                ":",
                round(
                    accuracy,
                    2
                ),
                "%",
                "- Matches:",
                len(subset)
            )

        else:

            print(
                level,
                ": No completed predictions"
            )


    print(
        "\n--- CHALLENGER ACCURACY BY CONFIDENCE ---"
    )

    for level in [
        "HIGH",
        "MEDIUM",
        "LOW"
    ]:

        subset = completed[
            completed[
                "challenger_confidence_level"
            ]
            ==
            level
        ]

        if len(subset) > 0:

            accuracy = (
                subset[
                    "challenger_correct"
                ]
                .astype(str)
                .str.lower()
                .eq("true")
                .mean()
                *
                100
            )

            print(
                level,
                ":",
                round(
                    accuracy,
                    2
                ),
                "%",
                "- Matches:",
                len(subset)
            )

        else:

            print(
                level,
                ": No completed predictions"
            )


# -------------------------------------------------
# Disagreement Cases
# -------------------------------------------------

print(
    "\n--- DISAGREEMENT CASES ---"
)

disagreements = df[
    ~agreement_mask
].copy()


if len(disagreements) > 0:

    disagreement_columns = [
        "prediction_time",
        "team1",
        "team2",
        "champion_winner",
        "challenger_winner",
        "probability_difference_pp",
        "result_status"
    ]

    print(
        disagreements[
            disagreement_columns
        ]
        .tail(20)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No champion/challenger "
        "disagreement cases yet."
    )


# -------------------------------------------------
# Recent Shadow Predictions
# -------------------------------------------------

print(
    "\n--- RECENT SHADOW PREDICTIONS ---"
)

recent_columns = [
    "prediction_time",
    "team1",
    "team2",
    "venue",
    "champion_winner",
    "challenger_winner",
    "models_agree",
    "probability_difference_pp",
    "result_status"
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


# -------------------------------------------------
# Promotion Readiness
# -------------------------------------------------

print(
    "\n================================"
)

print(
    "PROMOTION READINESS"
)

print(
    "================================"
)


minimum_completed = 30


print(
    "Minimum completed shadow "
    "predictions required:",
    minimum_completed
)

print(
    "Current completed predictions:",
    len(completed)
)


if (
    len(completed)
    <
    minimum_completed
):

    remaining = (
        minimum_completed
        -
        len(completed)
    )

    print(
        "Status:"
        " NOT READY FOR PROMOTION"
    )

    print(
        "More completed shadow "
        "predictions needed:",
        remaining
    )

    print(
        "Continue collecting "
        "ground-truth results."
    )


else:

    champion_accuracy = (
        completed[
            "champion_correct"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
        .mean()
        *
        100
    )

    challenger_accuracy = (
        completed[
            "challenger_correct"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
        .mean()
        *
        100
    )


    if (
        challenger_accuracy
        >
        champion_accuracy
    ):

        print(
            "Status:"
            " CHALLENGER ELIGIBLE"
            " FOR PROMOTION REVIEW"
        )

    elif (
        challenger_accuracy
        ==
        champion_accuracy
    ):

        print(
            "Status:"
            " MODELS CURRENTLY TIED"
        )

    else:

        print(
            "Status:"
            " KEEP CHAMPION"
        )


# -------------------------------------------------
# Important Note
# -------------------------------------------------

print(
    "\nImportant:"
)

print(
    "Promotion decisions should not "
    "be based on only one or two matches."
)

print(
    "Historical validation still favors "
    "the v1.1 challenger, but live shadow "
    "evidence must accumulate separately."
)