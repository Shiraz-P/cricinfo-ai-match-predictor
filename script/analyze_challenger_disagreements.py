import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "champion_challenger_v11_comparison.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "challenger_disagreement_analysis.csv"
)

# -------------------------------------------------
# Load comparison results
# -------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

print("========================================")
print("CHALLENGER DISAGREEMENT ANALYSIS")
print("========================================")

print("\nTotal compared matches:", len(df))

# -------------------------------------------------
# Select disagreements
# -------------------------------------------------

disagreements = df[
    df["models_disagree"] == True
].copy()

print(
    "Total disagreements:",
    len(disagreements)
)

# -------------------------------------------------
# Categorize disagreements
# -------------------------------------------------

def classify(row):

    if (
        row["challenger_correct"]
        and
        not row["champion_correct"]
    ):
        return "CHALLENGER_IMPROVEMENT"

    if (
        row["champion_correct"]
        and
        not row["challenger_correct"]
    ):
        return "CHALLENGER_REGRESSION"

    return "OTHER"


disagreements["category"] = (
    disagreements.apply(
        classify,
        axis=1
    )
)

# -------------------------------------------------
# Summary
# -------------------------------------------------

print("\n========================================")
print("SUMMARY")
print("========================================")

print(
    disagreements[
        "category"
    ].value_counts()
)

improvements = disagreements[
    disagreements["category"]
    ==
    "CHALLENGER_IMPROVEMENT"
].copy()

regressions = disagreements[
    disagreements["category"]
    ==
    "CHALLENGER_REGRESSION"
].copy()

print(
    "\nChallenger improvements:",
    len(improvements)
)

print(
    "Challenger regressions:",
    len(regressions)
)

print(
    "Net improvement:",
    len(improvements) - len(regressions)
)

# -------------------------------------------------
# Show Challenger improvements
# -------------------------------------------------

print("\n========================================")
print("CHALLENGER IMPROVEMENTS")
print("========================================")

if len(improvements) > 0:

    print(
        improvements[
            [
                "date",
                "team1",
                "team2",
                "winner",
                "champion_prediction",
                "challenger_prediction"
            ]
        ]
        .sort_values("date")
        .to_string(index=False)
    )

# -------------------------------------------------
# Show Challenger regressions
# -------------------------------------------------

print("\n========================================")
print("CHALLENGER REGRESSIONS")
print("========================================")

if len(regressions) > 0:

    print(
        regressions[
            [
                "date",
                "team1",
                "team2",
                "winner",
                "champion_prediction",
                "challenger_prediction"
            ]
        ]
        .sort_values("date")
        .to_string(index=False)
    )

# -------------------------------------------------
# Team-level analysis
#
# Count which teams appear most often in
# improvements and regressions.
# -------------------------------------------------

def team_counts(data):

    teams = pd.concat(
        [
            data["team1"],
            data["team2"]
        ],
        ignore_index=True
    )

    return teams.value_counts()


print("\n========================================")
print("TEAMS IN CHALLENGER IMPROVEMENTS")
print("========================================")

if len(improvements) > 0:

    print(
        team_counts(
            improvements
        ).head(15)
    )


print("\n========================================")
print("TEAMS IN CHALLENGER REGRESSIONS")
print("========================================")

if len(regressions) > 0:

    print(
        team_counts(
            regressions
        ).head(15)
    )

# -------------------------------------------------
# Monthly / temporal analysis
# -------------------------------------------------

disagreements["year_month"] = (
    disagreements[
        "date"
    ].dt.to_period("M").astype(str)
)

print("\n========================================")
print("DISAGREEMENTS BY MONTH")
print("========================================")

monthly = pd.crosstab(
    disagreements["year_month"],
    disagreements["category"]
)

print(monthly)

# -------------------------------------------------
# Winner analysis
# -------------------------------------------------

print("\n========================================")
print("WINNERS IN IMPROVEMENT CASES")
print("========================================")

if len(improvements) > 0:

    print(
        improvements[
            "winner"
        ]
        .value_counts()
        .head(15)
    )


print("\n========================================")
print("WINNERS IN REGRESSION CASES")
print("========================================")

if len(regressions) > 0:

    print(
        regressions[
            "winner"
        ]
        .value_counts()
        .head(15)
    )

# -------------------------------------------------
# Add simple interpretation field
# -------------------------------------------------

disagreements[
    "interpretation"
] = disagreements[
    "category"
].map({

    "CHALLENGER_IMPROVEMENT":
        "New pipeline corrected a Champion error",

    "CHALLENGER_REGRESSION":
        "New pipeline introduced an error",

    "OTHER":
        "Requires manual review"
})

# -------------------------------------------------
# Save analysis
# -------------------------------------------------

disagreements = disagreements.sort_values(
    "date"
).reset_index(drop=True)

disagreements.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("ANALYSIS SAVED")
print("========================================")

print(
    "Rows:",
    len(disagreements)
)

print(
    "File:",
    OUTPUT_FILE
)

print(
    "\nNo model was modified."
)