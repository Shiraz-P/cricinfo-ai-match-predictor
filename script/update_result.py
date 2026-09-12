import pandas as pd


# -------------------------------------------------
# Prediction log file
# -------------------------------------------------

log_file = r"K:\Python\Cricinfo_AI_Project\data\prediction_log.csv"

df = pd.read_csv(log_file)


# -------------------------------------------------
# Create monitoring columns if they do not exist
# -------------------------------------------------

if "actual_winner" not in df.columns:
    df["actual_winner"] = ""

if "correct_prediction" not in df.columns:
    df["correct_prediction"] = ""

if "result_status" not in df.columns:
    df["result_status"] = "PENDING"


# -------------------------------------------------
# Show existing predictions
# -------------------------------------------------

print("\n--- PREDICTION LOG ---")

display_columns = [
    "prediction_time",
    "team1",
    "team2",
    "venue",
    "predicted_winner",
    "confidence",
    "result_status"
]

print(
    df[display_columns]
    .tail(20)
    .to_string()
)


# -------------------------------------------------
# Ask which prediction to update
# -------------------------------------------------

while True:

    row_input = input(
        "\nEnter row number to update "
        "or type exit: "
    ).strip()

    if row_input.lower() == "exit":

        print("\nNo result updated.")
        exit()

    if row_input.isdigit():

        row_number = int(row_input)

        if row_number in df.index:
            break

    print("Invalid row number.")


# -------------------------------------------------
# Match details
# -------------------------------------------------

team1 = df.loc[
    row_number,
    "team1"
]

team2 = df.loc[
    row_number,
    "team2"
]

predicted_winner = df.loc[
    row_number,
    "predicted_winner"
]


print("\nSelected match:")

print(
    team1,
    "vs",
    team2
)

print(
    "Predicted Winner:",
    predicted_winner
)


# -------------------------------------------------
# Ask for actual winner
# -------------------------------------------------

while True:

    actual_winner = input(
        f"Enter Actual Winner "
        f"({team1}/{team2}) "
        f"or type skip: "
    ).strip()

    # ---------------------------------------------
    # Result not known yet
    # ---------------------------------------------

    if actual_winner.lower() == "skip":

        df.loc[
            row_number,
            "actual_winner"
        ] = ""

        df.loc[
            row_number,
            "correct_prediction"
        ] = ""

        df.loc[
            row_number,
            "result_status"
        ] = "PENDING"

        df.to_csv(
            log_file,
            index=False
        )

        print("\n--- RESULT NOT UPDATED ---")

        print(
            "Prediction remains PENDING."
        )

        print(
            "You can update this row later "
            "when the real result is known."
        )

        break


    # ---------------------------------------------
    # Team 1 won
    # ---------------------------------------------

    if actual_winner.lower() == team1.lower():

        actual_winner = team1

        break


    # ---------------------------------------------
    # Team 2 won
    # ---------------------------------------------

    elif actual_winner.lower() == team2.lower():

        actual_winner = team2

        break


    else:

        print(
            "Actual winner must be either",
            team1,
            "or",
            team2,
            "or type skip."
        )


# -------------------------------------------------
# Stop here if result is pending
# -------------------------------------------------

if actual_winner.lower() == "skip":

    exit()


# -------------------------------------------------
# Compare prediction with actual winner
# -------------------------------------------------

correct_prediction = (
    predicted_winner
    ==
    actual_winner
)


# -------------------------------------------------
# Update result
# -------------------------------------------------

df.loc[
    row_number,
    "actual_winner"
] = actual_winner

df.loc[
    row_number,
    "correct_prediction"
] = correct_prediction


if correct_prediction:

    df.loc[
        row_number,
        "result_status"
    ] = "CORRECT"

else:

    df.loc[
        row_number,
        "result_status"
    ] = "INCORRECT"


# -------------------------------------------------
# Save updated log
# -------------------------------------------------

df.to_csv(
    log_file,
    index=False
)


# -------------------------------------------------
# Result
# -------------------------------------------------

print("\n--- RESULT UPDATED ---")

print(
    "Predicted Winner:",
    predicted_winner
)

print(
    "Actual Winner:",
    actual_winner
)

print(
    "Prediction Correct:",
    correct_prediction
)

print(
    "Result Status:",
    df.loc[
        row_number,
        "result_status"
    ]
)


# -------------------------------------------------
# Monitoring Summary
# -------------------------------------------------

completed = df[
    df["result_status"].isin(
        [
            "CORRECT",
            "INCORRECT"
        ]
    )
].copy()


pending = df[
    df["result_status"] == "PENDING"
].copy()


print("\n--- MODEL MONITORING ---")

print(
    "Total logged predictions:",
    len(df)
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
# Live Accuracy
# -------------------------------------------------

if len(completed) > 0:

    monitoring_accuracy = (
        completed[
            "result_status"
        ]
        .eq("CORRECT")
        .mean()
    )

    print(
        "Live prediction accuracy:",
        round(
            monitoring_accuracy * 100,
            2
        ),
        "%"
    )

else:

    print(
        "Live prediction accuracy: "
        "Not available yet."
    )