import pandas as pd
from pathlib import Path

ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

LOG_FILE = (
    ROOT
    / "data"
    / "shadow_v12_log.csv"
)

print("========================================")
print("UPDATE SHADOW v1.2 RESULT")
print("========================================")


# -------------------------------------------------
# Load log
# -------------------------------------------------

df = pd.read_csv(LOG_FILE)

if len(df) == 0:
    print("Shadow log is empty.")
    raise SystemExit(0)


# -------------------------------------------------
# Normalize column types
# -------------------------------------------------

df["actual_winner"] = (
    df["actual_winner"]
    .astype("string")
)

df["result_status"] = (
    df["result_status"]
    .astype("string")
)

df["correct_prediction"] = (
    df["correct_prediction"]
    .astype("boolean")
)


# -------------------------------------------------
# Pending predictions
# -------------------------------------------------

pending = df[
    df["result_status"]
    .fillna("PENDING")
    .str.upper()
    ==
    "PENDING"
].copy()


if len(pending) == 0:

    print(
        "\nNo pending shadow predictions."
    )

    raise SystemExit(0)


print("\nPending predictions:\n")


for i, (index, row) in enumerate(
    pending.iterrows(),
    start=1
):

    print(
        f"{i}. "
        f"{row['team1']} vs {row['team2']} | "
        f"{row['venue']} | "
        f"Predicted: {row['predicted_winner']}"
    )


# -------------------------------------------------
# Select match
# -------------------------------------------------

choice = input(
    "\nSelect prediction number: "
).strip()


try:

    choice = int(choice)

except ValueError:

    print(
        "Invalid selection."
    )

    raise SystemExit(1)


if (
    choice < 1
    or
    choice > len(pending)
):

    print(
        "Selection out of range."
    )

    raise SystemExit(1)


selected_index = (
    pending.index[
        choice - 1
    ]
)

selected = df.loc[
    selected_index
]


team1 = str(
    selected["team1"]
)

team2 = str(
    selected["team2"]
)

predicted = str(
    selected["predicted_winner"]
)


print("\n========================================")
print("SELECTED MATCH")
print("========================================")

print(
    "Team 1:",
    team1
)

print(
    "Team 2:",
    team2
)

print(
    "Predicted winner:",
    predicted
)


# -------------------------------------------------
# Actual winner
# -------------------------------------------------

actual = input(
    "\nActual winner: "
).strip()


# Case-insensitive normalization
if actual.lower() == team1.lower():

    actual = team1

elif actual.lower() == team2.lower():

    actual = team2

else:

    print(
        "\nERROR:"
    )

    print(
        "Actual winner must match "
        "Team 1 or Team 2."
    )

    print(
        "Result NOT updated."
    )

    raise SystemExit(1)


# -------------------------------------------------
# Determine correctness
# -------------------------------------------------

correct = (
    actual == predicted
)


df.at[
    selected_index,
    "actual_winner"
] = actual


df.at[
    selected_index,
    "correct_prediction"
] = correct


df.at[
    selected_index,
    "result_status"
] = "COMPLETED"


# -------------------------------------------------
# Save
# -------------------------------------------------

df.to_csv(
    LOG_FILE,
    index=False
)


print("\n========================================")
print("RESULT UPDATED")
print("========================================")

print(
    "Actual winner:",
    actual
)

print(
    "Predicted winner:",
    predicted
)

print(
    "Correct prediction:",
    correct
)

print(
    "Status:",
    "COMPLETED"
)

print(
    "\nFile:",
    LOG_FILE
)