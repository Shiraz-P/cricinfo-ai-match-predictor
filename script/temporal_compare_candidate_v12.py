import pandas as pd
from pathlib import Path

ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

INPUT_FILE = (
    ROOT
    / "data"
    / "three_model_comparison.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "temporal_candidate_v12.csv"
)

print("========================================")
print("TEMPORAL VALIDATION - CANDIDATE v1.2")
print("========================================")

# -------------------------------------------------
# Load three-model comparison
# -------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["year_month"] = (
    df["date"]
    .dt.to_period("M")
    .astype(str)
)

print(
    "\nTotal matches:",
    len(df)
)

print(
    "Period:",
    df["date"].min(),
    "to",
    df["date"].max()
)


# -------------------------------------------------
# Monthly analysis
# -------------------------------------------------

rows = []

for month, group in df.groupby(
    "year_month",
    sort=True
):

    total = len(group)

    champion_correct = int(
        group["champion_correct"].sum()
    )

    candidate_correct = int(
        group["ablation_correct"].sum()
    )

    champion_accuracy = (
        champion_correct / total
    )

    candidate_accuracy = (
        candidate_correct / total
    )

    difference = (
        candidate_accuracy
        -
        champion_accuracy
    )

    candidate_improvements = int(
        (
            (~group["champion_correct"])
            &
            group["ablation_correct"]
        ).sum()
    )

    candidate_regressions = int(
        (
            group["champion_correct"]
            &
            (~group["ablation_correct"])
        ).sum()
    )

    net = (
        candidate_improvements
        -
        candidate_regressions
    )

    if difference > 0:

        monthly_winner = "CANDIDATE"

    elif difference < 0:

        monthly_winner = "CHAMPION"

    else:

        monthly_winner = "TIE"

    rows.append({

        "month":
            month,

        "matches":
            total,

        "champion_correct":
            champion_correct,

        "candidate_correct":
            candidate_correct,

        "champion_accuracy":
            champion_accuracy,

        "candidate_accuracy":
            candidate_accuracy,

        "difference_pp":
            difference * 100,

        "candidate_improvements":
            candidate_improvements,

        "candidate_regressions":
            candidate_regressions,

        "net_correct":
            net,

        "winner":
            monthly_winner
    })


result = pd.DataFrame(rows)


# -------------------------------------------------
# Display monthly results
# -------------------------------------------------

print("\n========================================")
print("MONTH-BY-MONTH RESULTS")
print("========================================")

display = result.copy()

display["champion_accuracy"] = (
    display["champion_accuracy"] * 100
).round(2)

display["candidate_accuracy"] = (
    display["candidate_accuracy"] * 100
).round(2)

display["difference_pp"] = (
    display["difference_pp"]
).round(2)

print(
    display.to_string(
        index=False
    )
)


# -------------------------------------------------
# Stability summary
# -------------------------------------------------

candidate_months = int(
    (
        result["winner"]
        ==
        "CANDIDATE"
    ).sum()
)

champion_months = int(
    (
        result["winner"]
        ==
        "CHAMPION"
    ).sum()
)

tie_months = int(
    (
        result["winner"]
        ==
        "TIE"
    ).sum()
)


print("\n========================================")
print("TEMPORAL STABILITY")
print("========================================")

print(
    "Candidate better months:",
    candidate_months
)

print(
    "Champion better months:",
    champion_months
)

print(
    "Tie months:",
    tie_months
)


# -------------------------------------------------
# Overall result
# -------------------------------------------------

total_matches = len(df)

champion_total_correct = int(
    df["champion_correct"].sum()
)

candidate_total_correct = int(
    df["ablation_correct"].sum()
)

champion_overall = (
    champion_total_correct
    /
    total_matches
)

candidate_overall = (
    candidate_total_correct
    /
    total_matches
)

overall_difference = (
    candidate_overall
    -
    champion_overall
)


print("\n========================================")
print("OVERALL")
print("========================================")

print(
    "Champion:",
    f"{champion_overall * 100:.2f}%"
)

print(
    "Candidate v1.2:",
    f"{candidate_overall * 100:.2f}%"
)

print(
    "Difference:",
    f"{overall_difference * 100:+.2f}",
    "percentage points"
)


# -------------------------------------------------
# Simple validation decision
# -------------------------------------------------

print("\n========================================")
print("TEMPORAL VALIDATION DECISION")
print("========================================")


if (
    candidate_overall > champion_overall
    and
    candidate_months > champion_months
):

    print(
        "Candidate improvement appears "
        "reasonably consistent across time."
    )

    print(
        "STATUS: Continue candidate validation."
    )

elif candidate_overall > champion_overall:

    print(
        "Candidate wins overall, but improvement "
        "is not consistently distributed over time."
    )

    print(
        "STATUS: More validation required."
    )

else:

    print(
        "Candidate does not outperform Champion "
        "over the evaluation period."
    )

    print(
        "STATUS: Keep Champion."
    )


# -------------------------------------------------
# Save
# -------------------------------------------------

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("RESULT SAVED")
print("========================================")

print(
    "File:",
    OUTPUT_FILE
)

print(
    "\nNo model was modified."
)