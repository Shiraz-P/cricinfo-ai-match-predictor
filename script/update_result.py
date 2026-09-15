import os
import pandas as pd


# ============================================================
# CRICINFO AI PROJECT
# Update Production Prediction Result
# ============================================================

PREDICTION_LOG = r"K:\Python\Cricinfo_AI_Project\data\prediction_log.csv"


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(PREDICTION_LOG):
    print("\nERROR: Prediction log file not found.")
    print("Expected file:")
    print(PREDICTION_LOG)
    raise SystemExit


# ============================================================
# LOAD PREDICTION LOG
# ============================================================

df = pd.read_csv(PREDICTION_LOG)


# ============================================================
# MAKE SURE REQUIRED RESULT COLUMNS EXIST
# ============================================================

# Text columns
for column in ["actual_winner", "result_status"]:

    if column not in df.columns:
        df[column] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        df[column] = df[column].astype("string")


# Boolean column
if "correct_prediction" not in df.columns:

    df["correct_prediction"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="boolean"
    )

else:

    df["correct_prediction"] = df["correct_prediction"].astype("boolean")


# ============================================================
# DISPLAY PREDICTION LOG
# ============================================================

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

# Only display columns that actually exist
display_columns = [
    column
    for column in display_columns
    if column in df.columns
]

print(
    df[display_columns].to_string()
)


# ============================================================
# ASK USER WHICH ROW TO UPDATE
# ============================================================

while True:

    user_input = input(
        "\nEnter row number to update or type exit: "
    ).strip()

    if user_input.lower() == "exit":
        print("\nNo changes made.")
        raise SystemExit

    try:
        row_number = int(user_input)

    except ValueError:
        print("Please enter a valid row number.")
        continue

    if row_number not in df.index:
        print("Invalid row number.")
        continue

    break


# ============================================================
# SELECT MATCH
# ============================================================

team1 = str(
    df.loc[row_number, "team1"]
).strip()

team2 = str(
    df.loc[row_number, "team2"]
).strip()

predicted_winner = str(
    df.loc[row_number, "predicted_winner"]
).strip()


print("\nSelected match:")
print(f"{team1} vs {team2}")

print(
    f"Predicted Winner: {predicted_winner}"
)


# ============================================================
# ASK FOR ACTUAL WINNER
# ============================================================

while True:

    actual_winner = input(
        f"Enter Actual Winner ({team1}/{team2}) "
        "or type skip: "
    ).strip()

    # --------------------------------------------------------
    # SKIP
    # --------------------------------------------------------

    if actual_winner.lower() == "skip":

        print("\nUpdate skipped.")
        raise SystemExit


    # --------------------------------------------------------
    # VALIDATE TEAM
    # --------------------------------------------------------

    if actual_winner.lower() == team1.lower():

        actual_winner = team1
        break

    elif actual_winner.lower() == team2.lower():

        actual_winner = team2
        break

    else:

        print(
            f"Invalid winner. Please enter "
            f"{team1} or {team2}."
        )


# ============================================================
# CHECK WHETHER MODEL PREDICTION WAS CORRECT
# ============================================================

correct_prediction = (
    predicted_winner.lower()
    ==
    actual_winner.lower()
)


# ============================================================
# UPDATE RESULT
# ============================================================

df.loc[
    row_number,
    "actual_winner"
] = actual_winner


df.loc[
    row_number,
    "correct_prediction"
] = correct_prediction


df.loc[
    row_number,
    "result_status"
] = "COMPLETED"


# ============================================================
# SAVE UPDATED CSV
# ============================================================

df.to_csv(
    PREDICTION_LOG,
    index=False
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n========================================")
print("PREDICTION RESULT UPDATED")
print("========================================")

print(f"Match             : {team1} vs {team2}")

print(
    f"Predicted Winner  : {predicted_winner}"
)

print(
    f"Actual Winner     : {actual_winner}"
)

print(
    f"Correct Prediction: {correct_prediction}"
)

print(
    "Result Status     : COMPLETED"
)


# ============================================================
# SIMPLE MODEL FEEDBACK
# ============================================================

if correct_prediction:

    print(
        "\nMODEL RESULT: CORRECT PREDICTION"
    )

else:

    print(
        "\nMODEL RESULT: INCORRECT PREDICTION"
    )


print("\nPrediction log updated successfully.")

print(
    f"File: {PREDICTION_LOG}"
)