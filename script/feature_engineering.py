import pandas as pd

file_path = r"K:\Python\Cricinfo_AI_Project\data\matches_clean.csv"

df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print("Dataset loaded:", df.shape)

print("\nFirst 5 chronological matches:")
print(df[["date", "team1", "team2", "winner"]].head())


# -------------------------------------------------
# Feature 1: Toss Winner is Team 1
# -------------------------------------------------

df["toss_winner_is_team1"] = (
    df["toss_winner"] == df["team1"]
).astype(int)

print("\nFeature 1: toss_winner_is_team1")

print(
    df[
        ["team1", "team2", "toss_winner", "toss_winner_is_team1"]
    ].head(10)
)


# -------------------------------------------------
# Features 2 & 3: Historical Win Rates
# -------------------------------------------------

team_matches = {}
team_wins = {}

team1_win_rates = []
team2_win_rates = []

for index, row in df.iterrows():

    team1 = row["team1"]
    team2 = row["team2"]
    winner = row["winner"]

    # Get historical records BEFORE this match
    team1_matches = team_matches.get(team1, 0)
    team2_matches = team_matches.get(team2, 0)

    team1_wins = team_wins.get(team1, 0)
    team2_wins = team_wins.get(team2, 0)

    # Calculate historical win rates
    team1_rate = team1_wins / team1_matches if team1_matches > 0 else 0.5
    team2_rate = team2_wins / team2_matches if team2_matches > 0 else 0.5

    team1_win_rates.append(team1_rate)
    team2_win_rates.append(team2_rate)

    # Update records AFTER calculating features
    team_matches[team1] = team1_matches + 1
    team_matches[team2] = team2_matches + 1

    if winner == team1:
        team_wins[team1] = team1_wins + 1

    elif winner == team2:
        team_wins[team2] = team2_wins + 1


df["team1_historical_win_rate"] = team1_win_rates
df["team2_historical_win_rate"] = team2_win_rates

print("\nFeatures 2 & 3: Historical Win Rates")

print(
    df[
        [
            "date",
            "team1",
            "team2",
            "team1_historical_win_rate",
            "team2_historical_win_rate",
            "winner"
        ]
    ].head(15)
)


# -------------------------------------------------
# Features 4 & 5: Recent Form (Last 5 Matches)
# -------------------------------------------------

recent_results = {}

team1_recent_rates = []
team2_recent_rates = []

for index, row in df.iterrows():

    team1 = row["team1"]
    team2 = row["team2"]
    winner = row["winner"]

    team1_history = recent_results.get(team1, [])
    team2_history = recent_results.get(team2, [])

    if len(team1_history) > 0:
        team1_recent_rate = sum(team1_history[-5:]) / len(team1_history[-5:])
    else:
        team1_recent_rate = 0.5

    if len(team2_history) > 0:
        team2_recent_rate = sum(team2_history[-5:]) / len(team2_history[-5:])
    else:
        team2_recent_rate = 0.5

    team1_recent_rates.append(team1_recent_rate)
    team2_recent_rates.append(team2_recent_rate)

    team1_result = 1 if winner == team1 else 0
    team2_result = 1 if winner == team2 else 0

    recent_results.setdefault(team1, []).append(team1_result)
    recent_results.setdefault(team2, []).append(team2_result)


df["team1_recent_win_rate"] = team1_recent_rates
df["team2_recent_win_rate"] = team2_recent_rates

print("\nFeatures 4 & 5: Recent Form (Last 5 Matches)")

print(
    df[
        [
            "date",
            "team1",
            "team2",
            "team1_recent_win_rate",
            "team2_recent_win_rate",
            "winner"
        ]
    ].head(15)
)


# -------------------------------------------------
# Features 6 & 7: Head-to-Head Historical Win Rates
# -------------------------------------------------

h2h_results = {}

team1_h2h_rates = []
team2_h2h_rates = []

for index, row in df.iterrows():

    team1 = row["team1"]
    team2 = row["team2"]
    winner = row["winner"]

    # Create same pair key regardless of team order
    pair = tuple(sorted([team1, team2]))

    previous_matches = h2h_results.get(pair, [])

    if len(previous_matches) == 0:
        team1_h2h_rate = 0.5
        team2_h2h_rate = 0.5

    else:
        total_previous = len(previous_matches)

        team1_previous_wins = previous_matches.count(team1)
        team2_previous_wins = previous_matches.count(team2)

        team1_h2h_rate = team1_previous_wins / total_previous
        team2_h2h_rate = team2_previous_wins / total_previous

    team1_h2h_rates.append(team1_h2h_rate)
    team2_h2h_rates.append(team2_h2h_rate)

    # Update history AFTER calculating current-match features
    h2h_results.setdefault(pair, []).append(winner)


df["team1_h2h_win_rate"] = team1_h2h_rates
df["team2_h2h_win_rate"] = team2_h2h_rates

print("\nFeatures 6 & 7: Head-to-Head Historical Win Rates")

print(
    df[
        [
            "date",
            "team1",
            "team2",
            "team1_h2h_win_rate",
            "team2_h2h_win_rate",
            "winner"
        ]
    ].head(15)
)


# -------------------------------------------------
# Save Final Feature-Engineered Dataset
# -------------------------------------------------

# -------------------------------------------------
# Features 8 & 9: Historical Venue Win Rates
# -------------------------------------------------

venue_team_matches = {}
venue_team_wins = {}

team1_venue_rates = []
team2_venue_rates = []

for index, row in df.iterrows():

    team1 = row["team1"]
    team2 = row["team2"]
    venue = row["venue"]
    winner = row["winner"]

    team1_key = (team1, venue)
    team2_key = (team2, venue)

    # Historical venue records BEFORE this match
    team1_matches_at_venue = venue_team_matches.get(
        team1_key,
        0
    )

    team2_matches_at_venue = venue_team_matches.get(
        team2_key,
        0
    )

    team1_wins_at_venue = venue_team_wins.get(
        team1_key,
        0
    )

    team2_wins_at_venue = venue_team_wins.get(
        team2_key,
        0
    )

    # Calculate venue win rate
    if team1_matches_at_venue > 0:
        team1_venue_rate = (
            team1_wins_at_venue /
            team1_matches_at_venue
        )
    else:
        team1_venue_rate = 0.5

    if team2_matches_at_venue > 0:
        team2_venue_rate = (
            team2_wins_at_venue /
            team2_matches_at_venue
        )
    else:
        team2_venue_rate = 0.5

    team1_venue_rates.append(
        team1_venue_rate
    )

    team2_venue_rates.append(
        team2_venue_rate
    )

    # Update venue records AFTER feature calculation
    venue_team_matches[team1_key] = (
        team1_matches_at_venue + 1
    )

    venue_team_matches[team2_key] = (
        team2_matches_at_venue + 1
    )

    if winner == team1:

        venue_team_wins[team1_key] = (
            team1_wins_at_venue + 1
        )

    elif winner == team2:

        venue_team_wins[team2_key] = (
            team2_wins_at_venue + 1
        )


df["team1_venue_win_rate"] = team1_venue_rates
df["team2_venue_win_rate"] = team2_venue_rates


print(
    "\nFeatures 8 & 9: Historical Venue Win Rates"
)

print(
    df[
        [
            "date",
            "team1",
            "team2",
            "venue",
            "team1_venue_win_rate",
            "team2_venue_win_rate",
            "winner"
        ]
    ].head(15)
)
feature_file = r"K:\Python\Cricinfo_AI_Project\data\matches_features.csv"

df.to_csv(feature_file, index=False)

print("\nFeature-engineered dataset created:")
print(feature_file)

print("\nFinal dataset shape:")
print(df.shape)