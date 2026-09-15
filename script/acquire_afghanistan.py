import pandas as pd
from pathlib import Path


# -------------------------------------------------
# Project paths
# -------------------------------------------------

PROJECT_ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

HISTORICAL_FILE = (
    PROJECT_ROOT
    / "afghanistan_history"
    / "Combined_Summary_T20_2005_to_May_11_2023.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "matches_afghanistan.csv"
)


# -------------------------------------------------
# Load Afghanistan historical source
# -------------------------------------------------

df = pd.read_csv(HISTORICAL_FILE)


# -------------------------------------------------
# Keep only Afghanistan matches
# -------------------------------------------------

afg = df[
    (df["Team 1"].str.strip() == "Afghanistan")
    |
    (df["Team 2"].str.strip() == "Afghanistan")
].copy()


# -------------------------------------------------
# Rename columns to our project format
# -------------------------------------------------

afg = afg.rename(
    columns={
        "Team 1": "team1",
        "Team 2": "team2",
        "Winner": "winner",
        "Ground": "venue",
        "Match Date": "date"
    }
)


# -------------------------------------------------
# Normalize team names
# -------------------------------------------------

team_mapping = {
    "U.A.E.": "United Arab Emirates",
    "P.N.G.": "Papua New Guinea",
    "USA": "United States of America"
}


for column in ["team1", "team2", "winner"]:

    afg[column] = (
        afg[column]
        .replace(team_mapping)
        .astype("string")
        .str.strip()
    )


# -------------------------------------------------
# Convert match date
# -------------------------------------------------

afg["date"] = pd.to_datetime(
    afg["date"],
    errors="coerce"
)


# -------------------------------------------------
# Create required project fields
# -------------------------------------------------

afg["match_type"] = "T20"

afg["season"] = (
    afg["date"]
    .dt.year
    .astype("Int64")
    .astype("string")
)


# Toss data is not available from this source.
# We keep it missing rather than inventing values.

afg["toss_winner"] = pd.NA
afg["toss_decision"] = pd.NA


# -------------------------------------------------
# Handle tied / no-result matches
# -------------------------------------------------

afg["winner"] = afg["winner"].replace(
    {
        "tied": "No Result",
        "-": "No Result"
    }
)


# -------------------------------------------------
# Select same schema as matches.csv
# -------------------------------------------------

columns = [
    "date",
    "team1",
    "team2",
    "venue",
    "toss_winner",
    "toss_decision",
    "match_type",
    "season",
    "winner"
]

afg = afg[columns]


# -------------------------------------------------
# Sort by date
# -------------------------------------------------

afg = afg.sort_values(
    by="date"
).reset_index(drop=True)


# -------------------------------------------------
# Save separate Afghanistan dataset
# -------------------------------------------------

afg.to_csv(
    OUTPUT_FILE,
    index=False
)


# -------------------------------------------------
# Validation output
# -------------------------------------------------

print("\n========================================")
print("AFGHANISTAN DATA ACQUISITION COMPLETE")
print("========================================")

print(
    "Total Afghanistan matches:",
    len(afg)
)

print(
    "Earliest match:",
    afg["date"].min()
)

print(
    "Latest match:",
    afg["date"].max()
)

print(
    "Missing toss winner:",
    afg["toss_winner"].isna().sum()
)

print(
    "Missing toss decision:",
    afg["toss_decision"].isna().sum()
)

print(
    "\nTeams found:"
)

teams = sorted(
    set(afg["team1"].dropna())
    |
    set(afg["team2"].dropna())
)

print(teams)

print(
    "\nOutput file:"
)

print(OUTPUT_FILE)

print(
    "\nLast 10 Afghanistan matches:"
)

print(
    afg.tail(10).to_string(
        index=False
    )
)