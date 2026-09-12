import pandas as pd


# -------------------------------------------------
# Shadow comparison log
# -------------------------------------------------

log_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\shadow_comparison_log.csv"
)


# -------------------------------------------------
# Load Shadow Log
# -------------------------------------------------

df = pd.read_csv(
    log_file
)


# -------------------------------------------------
# Fix / Normalize Column Data Types
#
# Blank CSV columns can be loaded by pandas
# as float64 because empty values become NaN.
#
# We convert the result-related columns to
# object dtype so they can safely store:
#
# actual_winner       -> text
# champion_correct    -> True / False
# challenger_correct  -> True / False
# result_status       -> text
# -------------------------------------------------

text_columns = [
    "actual_winner",
    "result_status"
]

for column in text_columns:

    if column not in df.columns:

        df[column] = ""

    df[column] = (
        df[column]
        .astype("object")
        .fillna("")
    )


boolean_result_columns = [
    "champion_correct",
    "challenger_correct"
]

for column in boolean_result_columns:

    if column not in df.columns:

        df[column] = ""

    df[column] = (
        df[column]
        .astype("object")
    )


# -------------------------------------------------
# Ensure Empty Result Status = PENDING
# -------------------------------------------------

df.loc[
    df["result_status"] == "",
    "result_status"
] = "PENDING"


# -------------------------------------------------
# Show Header
# -------------------------------------------------

print("\n================================")
print("SHADOW RESULT UPDATE")
print("================================")


# -------------------------------------------------
# Show Recent Shadow Predictions
# -------------------------------------------------

display_columns = [
    "prediction_time",
    "team1",
    "team2",
    "venue",
    "champion_winner",
    "challenger_winner",
    "models_agree",
    "result_status"
]


print(
    df[
        display_columns
    ]
    .tail(20)
    .to_string()
)


# -------------------------------------------------
# Select Row
# -------------------------------------------------

while True:

    row_input = input(
        "\nEnter row number to update "
        "or type exit: "
    ).strip()


    # ---------------------------------------------
    # Exit
    # ---------------------------------------------

    if row_input.lower() == "exit":

        print(
            "\nNo shadow result updated."
        )

        exit()


    # ---------------------------------------------
    # Validate numeric row
    # ---------------------------------------------

    if row_input.isdigit():

        row_number = int(
            row_input
        )

        if row_number in df.index:

            break


    print(
        "Invalid row number."
    )


# -------------------------------------------------
# Read Selected Match Details
# -------------------------------------------------

team1 = df.loc[
    row_number,
    "team1"
]

team2 = df.loc[
    row_number,
    "team2"
]

champion_winner = df.loc[
    row_number,
    "champion_winner"
]

challenger_winner = df.loc[
    row_number,
    "challenger_winner"
]

current_status = df.loc[
    row_number,
    "result_status"
]


# -------------------------------------------------
# Show Selected Match
# -------------------------------------------------

print(
    "\nSelected match:"
)

print(
    team1,
    "vs",
    team2
)

print(
    "Champion prediction:",
    champion_winner
)

print(
    "Challenger prediction:",
    challenger_winner
)

print(
    "Current Status:",
    current_status
)


# -------------------------------------------------
# If Already Completed
# -------------------------------------------------

if current_status == "COMPLETED":

    existing_actual_winner = df.loc[
        row_number,
        "actual_winner"
    ]

    print(
        "\nWARNING:"
    )

    print(
        "This shadow prediction has already "
        "been completed."
    )

    print(
        "Existing Actual Winner:",
        existing_actual_winner
    )

    overwrite = input(
        "Do you want to overwrite the result? "
        "(y/n): "
    ).strip().lower()

    if overwrite != "y":

        print(
            "\nNo changes made."
        )

        exit()


# -------------------------------------------------
# Actual Winner Input
# -------------------------------------------------

while True:

    actual_winner = input(
        f"Enter Actual Winner "
        f"({team1}/{team2}) "
        f"or type skip: "
    ).strip()


    # ---------------------------------------------
    # Skip / Keep Pending
    # ---------------------------------------------

    if actual_winner.lower() == "skip":

        df.loc[
            row_number,
            "actual_winner"
        ] = ""

        df.loc[
            row_number,
            "champion_correct"
        ] = ""

        df.loc[
            row_number,
            "challenger_correct"
        ] = ""

        df.loc[
            row_number,
            "result_status"
        ] = "PENDING"


        df.to_csv(
            log_file,
            index=False
        )


        print(
            "\n--- RESULT NOT UPDATED ---"
        )

        print(
            "Prediction remains PENDING."
        )

        print(
            "You can update this row later "
            "when the real result is known."
        )

        exit()


    # ---------------------------------------------
    # Team 1 Won
    # ---------------------------------------------

    if actual_winner.lower() == team1.lower():

        actual_winner = team1

        break


    # ---------------------------------------------
    # Team 2 Won
    # ---------------------------------------------

    elif actual_winner.lower() == team2.lower():

        actual_winner = team2

        break


    # ---------------------------------------------
    # Invalid Winner
    # ---------------------------------------------

    else:

        print(
            "Actual winner must be either",
            team1,
            "or",
            team2,
            "or type skip."
        )


# -------------------------------------------------
# Compare Champion Prediction With Ground Truth
# -------------------------------------------------

champion_correct = (
    champion_winner
    ==
    actual_winner
)


# -------------------------------------------------
# Compare Challenger Prediction With Ground Truth
# -------------------------------------------------

challenger_correct = (
    challenger_winner
    ==
    actual_winner
)


# -------------------------------------------------
# Update Row
# -------------------------------------------------

df.loc[
    row_number,
    "actual_winner"
] = actual_winner


df.loc[
    row_number,
    "champion_correct"
] = champion_correct


df.loc[
    row_number,
    "challenger_correct"
] = challenger_correct


df.loc[
    row_number,
    "result_status"
] = "COMPLETED"


# -------------------------------------------------
# Save Updated Shadow Log
# -------------------------------------------------

try:

    df.to_csv(
        log_file,
        index=False
    )

except PermissionError:

    print(
        "\nERROR:"
    )

    print(
        "Could not save shadow comparison log."
    )

    print(
        "Please close shadow_comparison_log.csv "
        "if it is open in Excel or another program."
    )

    exit()


# -------------------------------------------------
# Show Updated Result
# -------------------------------------------------

print(
    "\n--- SHADOW RESULT UPDATED ---"
)

print(
    "Actual Winner:",
    actual_winner
)

print(
    "Champion Prediction:",
    champion_winner
)

print(
    "Champion Correct:",
    champion_correct
)

print(
    "Challenger Prediction:",
    challenger_winner
)

print(
    "Challenger Correct:",
    challenger_correct
)

print(
    "Result Status: COMPLETED"
)


# -------------------------------------------------
# Select Completed Rows
# -------------------------------------------------

completed = df[
    df["result_status"]
    ==
    "COMPLETED"
].copy()


# -------------------------------------------------
# Select Pending Rows
# -------------------------------------------------

pending = df[
    df["result_status"]
    ==
    "PENDING"
].copy()


# -------------------------------------------------
# Shadow Monitoring Summary
# -------------------------------------------------

print(
    "\n================================"
)

print(
    "SHADOW MONITORING SUMMARY"
)

print(
    "================================"
)


print(
    "Total shadow predictions:",
    len(df)
)

print(
    "Completed:",
    len(completed)
)

print(
    "Pending:",
    len(pending)
)


# -------------------------------------------------
# Live Champion / Challenger Accuracy
# -------------------------------------------------

if len(completed) > 0:

    champion_accuracy = (
        completed[
            "champion_correct"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
        .mean()
    )


    challenger_accuracy = (
        completed[
            "challenger_correct"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
        .mean()
    )


    print(
        "Champion Live Accuracy:",
        round(
            champion_accuracy
            * 100,
            2
        ),
        "%"
    )


    print(
        "Challenger Live Accuracy:",
        round(
            challenger_accuracy
            * 100,
            2
        ),
        "%"
    )


    # ---------------------------------------------
    # Current Shadow Leader
    # ---------------------------------------------

    if (
        challenger_accuracy
        >
        champion_accuracy
    ):

        print(
            "Current shadow leader: CHALLENGER"
        )


    elif (
        champion_accuracy
        >
        challenger_accuracy
    ):

        print(
            "Current shadow leader: CHAMPION"
        )


    else:

        print(
            "Current shadow result: TIED"
        )


else:

    print(
        "Live shadow accuracy: "
        "Not available yet."
    )


# -------------------------------------------------
# Agreement / Disagreement Summary
# -------------------------------------------------

print(
    "\n--- MODEL AGREEMENT SUMMARY ---"
)

agreement_count = (
    df[
        "models_agree"
    ]
    .astype(str)
    .str.lower()
    .eq("true")
    .sum()
)


disagreement_count = (
    len(df)
    -
    agreement_count
)


print(
    "Models agreed:",
    agreement_count
)

print(
    "Models disagreed:",
    disagreement_count
)


# -------------------------------------------------
# Completed Agreement Cases
# -------------------------------------------------

if len(completed) > 0:

    completed_agreements = completed[
        completed[
            "models_agree"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
    ]


    completed_disagreements = completed[
        ~completed[
            "models_agree"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
    ]


    print(
        "Completed agreement cases:",
        len(
            completed_agreements
        )
    )

    print(
        "Completed disagreement cases:",
        len(
            completed_disagreements
        )
    )


# -------------------------------------------------
# Save Confirmation
# -------------------------------------------------

print(
    "\nShadow comparison log updated:"
)

print(
    log_file
)