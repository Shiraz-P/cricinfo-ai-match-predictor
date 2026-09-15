import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"K:\Python\Cricinfo_AI_Project")
DATA_DIR = PROJECT_ROOT / "data"

CORE_FILE = DATA_DIR / "matches.csv"
HIST_FILE = DATA_DIR / "matches_afghanistan.csv"
RECENT_FILE = DATA_DIR / "matches_afghanistan_final.csv"

OUTPUT_FILE = DATA_DIR / "matches_candidate_v2.csv"

# -------------------------------------------------
# Load datasets
# -------------------------------------------------

core = pd.read_csv(CORE_FILE)
historical = pd.read_csv(HIST_FILE)
recent = pd.read_csv(RECENT_FILE)

print("========================================")
print("BUILD CANDIDATE DATASET V2")
print("========================================")

print("\nCore rows:", len(core))
print("Historical Afghanistan:", len(historical))
print("Recent Afghanistan:", len(recent))

# -------------------------------------------------
# Required model columns
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

core = core[columns].copy()
historical = historical[columns].copy()
recent = recent[columns].copy()

# -------------------------------------------------
# Normalize dates
# -------------------------------------------------

for df in [core, historical, recent]:
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

# -------------------------------------------------
# Normalize team names
# -------------------------------------------------

TEAM_MAP = {
    "U.A.E.": "United Arab Emirates",
    "UAE": "United Arab Emirates",
    "P.N.G.": "Papua New Guinea",
    "PNG": "Papua New Guinea",
    "AFG": "Afghanistan",
    "BAN": "Bangladesh"
}

for df in [core, historical, recent]:

    for column in [
        "team1",
        "team2",
        "winner",
        "toss_winner"
    ]:

        df[column] = df[column].replace(TEAM_MAP)

# -------------------------------------------------
# Validate supplemental Afghanistan data
# -------------------------------------------------

supplemental = pd.concat(
    [historical, recent],
    ignore_index=True
)

print(
    "\nTotal Afghanistan supplemental rows:",
    len(supplemental)
)

# -------------------------------------------------
# Create comparison key
#
# IMPORTANT:
# This key is ONLY for comparing supplemental
# Afghanistan data with the core dataset.
#
# We DO NOT use it to deduplicate core itself.
# -------------------------------------------------

def create_key(df):

    result = df.copy()

    result["_team_low"] = result[
        ["team1", "team2"]
    ].min(axis=1)

    result["_team_high"] = result[
        ["team1", "team2"]
    ].max(axis=1)

    result["_match_key"] = (
        result["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + result["_team_low"].astype(str)
        + "|"
        + result["_team_high"].astype(str)
    )

    return result


core_check = create_key(core)
supp_check = create_key(supplemental)

# -------------------------------------------------
# Identify supplemental records already represented
# in core.
#
# Do NOT remove anything from core.
# -------------------------------------------------

core_keys = set(
    core_check["_match_key"]
)

supp_check["_exists_in_core"] = (
    supp_check["_match_key"].isin(core_keys)
)

overlap = supp_check[
    supp_check["_exists_in_core"]
].copy()

new_afghanistan = supp_check[
    ~supp_check["_exists_in_core"]
].copy()

print(
    "\nSupplemental rows matching core:",
    len(overlap)
)

print(
    "Supplemental rows NOT in core:",
    len(new_afghanistan)
)

# -------------------------------------------------
# Show overlaps for validation
# -------------------------------------------------

if not overlap.empty:

    print("\nAfghanistan overlap with core:")

    print(
        overlap[
            [
                "date",
                "team1",
                "team2",
                "winner"
            ]
        ].to_string(index=False)
    )

# -------------------------------------------------
# Check duplicate keys INSIDE supplemental data.
#
# We report these instead of silently deleting them.
# -------------------------------------------------

supp_duplicates = supp_check[
    supp_check.duplicated(
        subset=["_match_key"],
        keep=False
    )
]

print(
    "\nDuplicate supplemental rows:",
    len(supp_duplicates)
)

print(
    "Duplicate supplemental match keys:",
    supp_duplicates["_match_key"].nunique()
)

if not supp_duplicates.empty:

    print("\nWARNING - Supplemental duplicates:")

    print(
        supp_duplicates[
            [
                "date",
                "team1",
                "team2",
                "winner"
            ]
        ].sort_values(
            "date"
        ).to_string(index=False)
    )

# -------------------------------------------------
# Remove temporary columns from NEW Afghanistan data
# -------------------------------------------------

new_afghanistan = new_afghanistan[
    columns
].copy()

# -------------------------------------------------
# IMPORTANT:
#
# Preserve ALL original 3,539 core records.
# Add only Afghanistan records not represented
# in the core dataset.
# -------------------------------------------------

candidate = pd.concat(
    [
        core,
        new_afghanistan
    ],
    ignore_index=True
)

candidate = candidate.sort_values(
    "date"
).reset_index(drop=True)

# -------------------------------------------------
# Validation
# -------------------------------------------------

print("\n========================================")
print("VALIDATION")
print("========================================")

print(
    "\nOriginal core rows preserved:",
    len(core)
)

print(
    "New Afghanistan rows added:",
    len(new_afghanistan)
)

print(
    "Final candidate rows:",
    len(candidate)
)

expected = (
    len(core)
    + len(new_afghanistan)
)

print(
    "Expected candidate rows:",
    expected
)

print(
    "Row count validation:",
    len(candidate) == expected
)

# -------------------------------------------------
# Afghanistan validation
# -------------------------------------------------

afg = candidate[
    (candidate["team1"] == "Afghanistan")
    |
    (candidate["team2"] == "Afghanistan")
]

print(
    "\nAfghanistan matches in candidate:",
    len(afg)
)

print(
    "Afghanistan earliest:",
    afg["date"].min()
)

print(
    "Afghanistan latest:",
    afg["date"].max()
)

print("\nAfghanistan matches per year:")

print(
    afg["date"]
    .dt.year
    .value_counts()
    .sort_index()
)

# -------------------------------------------------
# Invalid fixture safety check
# -------------------------------------------------

invalid_dates = pd.to_datetime([
    "2025-11-17",
    "2025-11-23",
    "2025-11-25"
])

bad = afg[
    afg["date"].isin(invalid_dates)
]

print(
    "\nInvalid Nov-2025 Afghanistan fixtures:",
    len(bad)
)

# -------------------------------------------------
# Missing toss check
# -------------------------------------------------

print(
    "Candidate rows with missing toss:",
    candidate["toss_winner"].isna().sum()
)

# -------------------------------------------------
# Save
# -------------------------------------------------

candidate.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("CANDIDATE V2 CREATED")
print("========================================")

print("Saved:", OUTPUT_FILE)

print(
    "\nChampion v1.0 files were NOT modified."
)