import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"K:\Python\Cricinfo_AI_Project")
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "matches_candidate_v2.csv"
OUTPUT_FILE = DATA_DIR / "matches_candidate_features.csv"

# -------------------------------------------------
# Load candidate dataset
# -------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.sort_values("date").reset_index(drop=True)

print("========================================")
print("CANDIDATE FEATURE ENGINEERING")
print("========================================")

print("\nInput matches:", len(df))

# -------------------------------------------------
# Target
# -------------------------------------------------

df["team1_won"] = (
    df["winner"] == df["team1"]
).astype(int)

# -------------------------------------------------
# FEATURE 1
# Toss winner is team1
#
# Known:
# team1 won toss  = 1.0
# team1 lost toss = 0.0
#
# Unknown:
# neutral value   = 0.5
# -------------------------------------------------

df["toss_winner_is_team1"] = 0.5

known_toss = df["toss_winner"].notna()

df.loc[
    known_toss,
    "toss_winner_is_team1"
] = (
    df.loc[known_toss, "toss_winner"]
    ==
    df.loc[known_toss, "team1"]
).astype(float)

# -------------------------------------------------
# FEATURE 2
# Toss data known
#
# 1 = toss information available
# 0 = toss information unavailable
# -------------------------------------------------

df["toss_data_known"] = (
    df["toss_winner"].notna()
).astype(int)

# -------------------------------------------------
# Historical structures
# -------------------------------------------------

team_stats = {}
recent_results = {}
h2h_stats = {}
venue_stats = {}

team1_historical = []
team2_historical = []

team1_recent = []
team2_recent = []

team1_h2h = []
team2_h2h = []

team1_venue = []
team2_venue = []

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def historical_win_rate(team):

    stats = team_stats.get(
        team,
        {"wins": 0, "matches": 0}
    )

    if stats["matches"] == 0:
        return 0.5

    return stats["wins"] / stats["matches"]


def recent_win_rate(team):

    results = recent_results.get(team, [])

    if len(results) == 0:
        return 0.5

    last_five = results[-5:]

    return sum(last_five) / len(last_five)


def h2h_win_rate(team, opponent):

    key = tuple(sorted([team, opponent]))

    stats = h2h_stats.get(
        key,
        {
            team: {"wins": 0, "matches": 0},
            opponent: {"wins": 0, "matches": 0}
        }
    )

    if team not in stats:
        return 0.5

    if stats[team]["matches"] == 0:
        return 0.5

    return (
        stats[team]["wins"]
        /
        stats[team]["matches"]
    )


def venue_win_rate(team, venue):

    key = (team, venue)

    stats = venue_stats.get(
        key,
        {"wins": 0, "matches": 0}
    )

    if stats["matches"] == 0:
        return 0.5

    return stats["wins"] / stats["matches"]


# -------------------------------------------------
# Process chronologically
#
# IMPORTANT:
# Features are calculated BEFORE current match
# result is added to history.
#
# This prevents DATA LEAKAGE.
# -------------------------------------------------

for _, row in df.iterrows():

    team1 = row["team1"]
    team2 = row["team2"]
    winner = row["winner"]
    venue = row["venue"]

    # Historical win rate
    team1_historical.append(
        historical_win_rate(team1)
    )

    team2_historical.append(
        historical_win_rate(team2)
    )

    # Recent form
    team1_recent.append(
        recent_win_rate(team1)
    )

    team2_recent.append(
        recent_win_rate(team2)
    )

    # Head-to-head
    team1_h2h.append(
        h2h_win_rate(team1, team2)
    )

    team2_h2h.append(
        h2h_win_rate(team2, team1)
    )

    # Venue
    team1_venue.append(
        venue_win_rate(team1, venue)
    )

    team2_venue.append(
        venue_win_rate(team2, venue)
    )

    # ---------------------------------------------
    # Update team historical statistics
    # ---------------------------------------------

    for team in [team1, team2]:

        if team not in team_stats:
            team_stats[team] = {
                "wins": 0,
                "matches": 0
            }

        team_stats[team]["matches"] += 1

    if winner == team1:
        team_stats[team1]["wins"] += 1

    elif winner == team2:
        team_stats[team2]["wins"] += 1

    # ---------------------------------------------
    # Update recent results
    # ---------------------------------------------

    if team1 not in recent_results:
        recent_results[team1] = []

    if team2 not in recent_results:
        recent_results[team2] = []

    if winner == team1:

        recent_results[team1].append(1)
        recent_results[team2].append(0)

    elif winner == team2:

        recent_results[team1].append(0)
        recent_results[team2].append(1)

    else:

        # No Result / Tie:
        # do not count as a win for either side.
        recent_results[team1].append(0)
        recent_results[team2].append(0)

    # ---------------------------------------------
    # Update H2H statistics
    # ---------------------------------------------

    h2h_key = tuple(
        sorted([team1, team2])
    )

    if h2h_key not in h2h_stats:

        h2h_stats[h2h_key] = {
            team1: {
                "wins": 0,
                "matches": 0
            },
            team2: {
                "wins": 0,
                "matches": 0
            }
        }

    for team in [team1, team2]:

        if team not in h2h_stats[h2h_key]:

            h2h_stats[h2h_key][team] = {
                "wins": 0,
                "matches": 0
            }

        h2h_stats[h2h_key][team]["matches"] += 1

    if winner == team1:
        h2h_stats[h2h_key][team1]["wins"] += 1

    elif winner == team2:
        h2h_stats[h2h_key][team2]["wins"] += 1

    # ---------------------------------------------
    # Update venue statistics
    # ---------------------------------------------

    for team in [team1, team2]:

        venue_key = (team, venue)

        if venue_key not in venue_stats:

            venue_stats[venue_key] = {
                "wins": 0,
                "matches": 0
            }

        venue_stats[venue_key]["matches"] += 1

    if winner == team1:

        venue_stats[
            (team1, venue)
        ]["wins"] += 1

    elif winner == team2:

        venue_stats[
            (team2, venue)
        ]["wins"] += 1


# -------------------------------------------------
# Add calculated features
# -------------------------------------------------

df["team1_historical_win_rate"] = team1_historical
df["team2_historical_win_rate"] = team2_historical

df["team1_recent_win_rate"] = team1_recent
df["team2_recent_win_rate"] = team2_recent

df["team1_h2h_win_rate"] = team1_h2h
df["team2_h2h_win_rate"] = team2_h2h

df["team1_venue_win_rate"] = team1_venue
df["team2_venue_win_rate"] = team2_venue

# -------------------------------------------------
# Challenger feature list
# -------------------------------------------------

features = [
    "toss_winner_is_team1",
    "toss_data_known",
    "team1_historical_win_rate",
    "team2_historical_win_rate",
    "team1_recent_win_rate",
    "team2_recent_win_rate",
    "team1_h2h_win_rate",
    "team2_h2h_win_rate",
    "team1_venue_win_rate",
    "team2_venue_win_rate"
]

# -------------------------------------------------
# Validation
# -------------------------------------------------

print("\n========================================")
print("FEATURE VALIDATION")
print("========================================")

print("\nNumber of features:", len(features))

print(
    "Rows with known toss:",
    int(df["toss_data_known"].sum())
)

print(
    "Rows with unknown toss:",
    int((df["toss_data_known"] == 0).sum())
)

unknown = df[
    df["toss_data_known"] == 0
]

print(
    "Unknown toss rows using neutral 0.5:",
    int(
        (
            unknown["toss_winner_is_team1"]
            == 0.5
        ).sum()
    )
)

print(
    "\nMissing values inside model features:",
    int(df[features].isna().sum().sum())
)

print("\nFeature names:")

for number, feature in enumerate(
    features,
    start=1
):
    print(number, feature)

# -------------------------------------------------
# Afghanistan sample
# -------------------------------------------------

afg = df[
    (df["team1"] == "Afghanistan")
    |
    (df["team2"] == "Afghanistan")
]

print("\nAfghanistan feature rows:", len(afg))

print("\nFirst 5 Afghanistan feature rows:")

print(
    afg[
        [
            "date",
            "team1",
            "team2",
            "toss_winner_is_team1",
            "toss_data_known",
            "team1_historical_win_rate",
            "team2_historical_win_rate",
            "team1_recent_win_rate",
            "team2_recent_win_rate",
            "winner",
            "team1_won"
        ]
    ]
    .head()
    .to_string(index=False)
)

# -------------------------------------------------
# Save
# -------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("CANDIDATE FEATURES CREATED")
print("========================================")

print("Rows:", len(df))
print("Features:", len(features))
print("Saved:", OUTPUT_FILE)

print(
    "\nChampion v1.0 feature/model files were NOT modified."
)