from difflib import get_close_matches
from datetime import datetime

import os
import pandas as pd
import joblib


# -------------------------------------------------
# Load feature-engineered dataset
# -------------------------------------------------

file_path = r"K:\Python\Cricinfo_AI_Project\data\matches_features.csv"

df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# -------------------------------------------------
# Predictor Feature Schema
# -------------------------------------------------

features = [
    "toss_winner_is_team1",
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
# Load Pre-Trained Model Package
# -------------------------------------------------

model_path = r"K:\Python\Cricinfo_AI_Project\model\cricket_model.pkl"

model_package = joblib.load(model_path)

model = model_package["model"]
saved_features = model_package["features"]

print("\n--- MODEL INFORMATION ---")

print("Trained model package loaded successfully.")
print("Model version:", model_package["model_version"])
print("Model type:", model_package["model_type"])
print("Saved test accuracy:", model_package["test_accuracy"], "%")
print("Saved feature count:", model_package["feature_count"])


# -------------------------------------------------
# Verify Feature Schema
# -------------------------------------------------

if features != saved_features:

    print("\nERROR: Feature schema mismatch.")

    print("Predictor features:")
    print(features)

    print("\nSaved model features:")
    print(saved_features)

    raise ValueError(
        "Prediction stopped because predictor features "
        "do not match saved model features."
    )

print(
    "Feature schema verified:",
    len(saved_features),
    "features"
)


# -------------------------------------------------
# Build official team list
# -------------------------------------------------

all_teams = sorted(
    set(df["team1"]).union(
        set(df["team2"])
    )
)


# -------------------------------------------------
# Function: normalize team name
# -------------------------------------------------

def normalize_team(user_input):

    normalized_input = (
        user_input
        .strip()
        .lower()
        .replace(" ", "")
    )

    # First try exact normalized match
    for team in all_teams:

        normalized_team = (
            team
            .lower()
            .replace(" ", "")
        )

        if normalized_team == normalized_input:
            return team

    # Fuzzy matching for possible typo
    normalized_team_map = {
        team.lower().replace(" ", ""): team
        for team in all_teams
    }

    possible_matches = get_close_matches(
        normalized_input,
        normalized_team_map.keys(),
        n=1,
        cutoff=0.75
    )

    if len(possible_matches) == 1:

        suggested_team = normalized_team_map[
            possible_matches[0]
        ]

        answer = input(
            f"Team not found. Did you mean "
            f"{suggested_team}? (y/n): "
        ).strip().lower()

        if answer == "y":
            return suggested_team

    return None


# -------------------------------------------------
# Generic text normalization
# Used mainly for venue matching
# -------------------------------------------------

def normalize_text(value):

    return (
        value
        .strip()
        .lower()
        .replace(" ", "")
        .replace("'", "")
        .replace("-", "")
        .replace(".", "")
        .replace(",", "")
    )


# -------------------------------------------------
# Function: resolve venue
# Exact -> Partial -> Fuzzy
# -------------------------------------------------

def resolve_venue(user_input):

    normalized_input = normalize_text(
        user_input
    )

    all_venues = sorted(
        df["venue"].dropna().unique()
    )

    # -------------------------------------------------
    # 1. Exact normalized match
    # Example:
    # lords -> Lord's
    # -------------------------------------------------

    for venue in all_venues:

        if normalize_text(venue) == normalized_input:
            return venue


    # -------------------------------------------------
    # 2. Partial normalized match
    # Example:
    # sydney -> Sydney Cricket Ground
    # -------------------------------------------------

    matches = [
        venue
        for venue in all_venues
        if normalized_input in normalize_text(venue)
    ]


    if len(matches) == 1:

        return matches[0]


    elif len(matches) > 1:

        print(
            "\nMultiple venues matched your input:"
        )

        for number, venue in enumerate(
            matches,
            start=1
        ):

            print(
                number,
                "-",
                venue
            )

        while True:

            choice = input(
                "Select venue number: "
            ).strip()

            if choice.isdigit():

                choice_number = int(choice)

                if (
                    1
                    <= choice_number
                    <= len(matches)
                ):

                    return matches[
                        choice_number - 1
                    ]

            print(
                "Invalid choice. Please enter "
                "one of the venue numbers shown."
            )


    # -------------------------------------------------
    # 3. Fuzzy venue matching
    # Example:
    # lrod -> Lord's
    # -------------------------------------------------

    normalized_venue_map = {
        normalize_text(venue): venue
        for venue in all_venues
    }

    possible_matches = get_close_matches(
        normalized_input,
        normalized_venue_map.keys(),
        n=3,
        cutoff=0.70
    )


    if len(possible_matches) == 1:

        suggested_venue = normalized_venue_map[
            possible_matches[0]
        ]

        answer = input(
            f"Venue not found. Did you mean "
            f"{suggested_venue}? (y/n): "
        ).strip().lower()

        if answer == "y":
            return suggested_venue


    elif len(possible_matches) > 1:

        print(
            "\nPossible venue matches:"
        )

        suggested_venues = [
            normalized_venue_map[item]
            for item in possible_matches
        ]

        for number, venue in enumerate(
            suggested_venues,
            start=1
        ):

            print(
                number,
                "-",
                venue
            )

        while True:

            choice = input(
                "Select venue number "
                "or 0 to try again: "
            ).strip()

            if choice == "0":
                return None

            if choice.isdigit():

                choice_number = int(choice)

                if (
                    1
                    <= choice_number
                    <= len(suggested_venues)
                ):

                    return suggested_venues[
                        choice_number - 1
                    ]

            print(
                "Invalid choice."
            )


    return None


# -------------------------------------------------
# Function: overall historical win rate
# -------------------------------------------------

def historical_win_rate(team):

    team_matches = df[
        (df["team1"] == team)
        |
        (df["team2"] == team)
    ]

    if len(team_matches) == 0:
        return 0.5

    wins = len(
        team_matches[
            team_matches["winner"] == team
        ]
    )

    return wins / len(team_matches)


# -------------------------------------------------
# Function: recent form - last 5 matches
# -------------------------------------------------

def recent_win_rate(team):

    team_matches = df[
        (df["team1"] == team)
        |
        (df["team2"] == team)
    ].sort_values(
        "date"
    )

    last_five = team_matches.tail(5)

    if len(last_five) == 0:
        return 0.5

    wins = len(
        last_five[
            last_five["winner"] == team
        ]
    )

    return wins / len(last_five)


# -------------------------------------------------
# Function: head-to-head win rates
# -------------------------------------------------

def head_to_head(team1, team2):

    h2h_matches = df[
        (
            (df["team1"] == team1)
            &
            (df["team2"] == team2)
        )
        |
        (
            (df["team1"] == team2)
            &
            (df["team2"] == team1)
        )
    ]

    if len(h2h_matches) == 0:
        return 0.5, 0.5

    team1_wins = len(
        h2h_matches[
            h2h_matches["winner"] == team1
        ]
    )

    team2_wins = len(
        h2h_matches[
            h2h_matches["winner"] == team2
        ]
    )

    total = len(
        h2h_matches
    )

    return (
        team1_wins / total,
        team2_wins / total
    )


# -------------------------------------------------
# Function: venue win rate
# -------------------------------------------------

def venue_win_rate(team, venue):

    venue_matches = df[
        (
            (df["team1"] == team)
            |
            (df["team2"] == team)
        )
        &
        (df["venue"] == venue)
    ]

    if len(venue_matches) == 0:
        return 0.5

    wins = len(
        venue_matches[
            venue_matches["winner"] == team
        ]
    )

    return wins / len(venue_matches)


# -------------------------------------------------
# User Input
# -------------------------------------------------

print(
    "\n--- CRICKET MATCH PREDICTOR ---"
)


# -------------------------------------------------
# Team 1 Input Validation
# -------------------------------------------------

while True:

    team1_input = input(
        "Enter Team 1: "
    )

    team1 = normalize_team(
        team1_input
    )

    if team1 is not None:
        break

    print(
        "Team not found in dataset. "
        "Please enter a valid team."
    )


# -------------------------------------------------
# Team 2 Input Validation
# -------------------------------------------------

while True:

    team2_input = input(
        "Enter Team 2: "
    )

    team2 = normalize_team(
        team2_input
    )

    if team2 is None:

        print(
            "Team not found in dataset. "
            "Please enter a valid team."
        )

        continue

    if team2 == team1:

        print(
            "Team 2 must be different "
            "from Team 1."
        )

        continue

    break


# -------------------------------------------------
# Venue Input Validation
# -------------------------------------------------

while True:

    venue_input = input(
        "Enter Venue or City: "
    )

    venue = resolve_venue(
        venue_input
    )

    if venue is not None:
        break

    print(
        "Venue not found. "
        "Please try another venue or city name."
    )


# -------------------------------------------------
# Toss Winner Validation
# -------------------------------------------------

while True:

    toss_input = input(
        "Enter Toss Winner: "
    )

    toss_winner = normalize_team(
        toss_input
    )

    if toss_winner not in [
        team1,
        team2
    ]:

        print(
            "Invalid toss winner."
        )

        print(
            "Toss winner must be either",
            team1,
            "or",
            team2
        )

        continue

    break


# -------------------------------------------------
# Calculate features automatically
# -------------------------------------------------

toss_winner_is_team1 = (
    1
    if toss_winner == team1
    else 0
)

team1_historical = historical_win_rate(
    team1
)

team2_historical = historical_win_rate(
    team2
)

team1_recent = recent_win_rate(
    team1
)

team2_recent = recent_win_rate(
    team2
)

team1_h2h, team2_h2h = head_to_head(
    team1,
    team2
)

team1_venue = venue_win_rate(
    team1,
    venue
)

team2_venue = venue_win_rate(
    team2,
    venue
)


# -------------------------------------------------
# Build prediction input
# -------------------------------------------------

new_match = pd.DataFrame([{
    "toss_winner_is_team1":
        toss_winner_is_team1,

    "team1_historical_win_rate":
        team1_historical,

    "team2_historical_win_rate":
        team2_historical,

    "team1_recent_win_rate":
        team1_recent,

    "team2_recent_win_rate":
        team2_recent,

    "team1_h2h_win_rate":
        team1_h2h,

    "team2_h2h_win_rate":
        team2_h2h,

    "team1_venue_win_rate":
        team1_venue,

    "team2_venue_win_rate":
        team2_venue
}])


# -------------------------------------------------
# Final Feature Safety Check
# -------------------------------------------------

if list(new_match.columns) != saved_features:

    raise ValueError(
        "Prediction input features do not match "
        "the saved model feature schema."
    )


# -------------------------------------------------
# Prediction
# -------------------------------------------------

prediction = model.predict(
    new_match
)[0]

probabilities = model.predict_proba(
    new_match
)[0]

team1_probability = probabilities[1]
team2_probability = probabilities[0]


# -------------------------------------------------
# Confidence Level
# -------------------------------------------------

confidence = max(
    team1_probability,
    team2_probability
)

if confidence >= 0.70:

    confidence_level = "HIGH"

elif confidence >= 0.60:

    confidence_level = "MEDIUM"

else:

    confidence_level = "LOW"


# -------------------------------------------------
# Output
# -------------------------------------------------

print(
    "\n--- MATCH DETAILS ---"
)

print(
    "Team 1:",
    team1
)

print(
    "Team 2:",
    team2
)

print(
    "Venue:",
    venue
)

print(
    "Toss Winner:",
    toss_winner
)


print(
    "\n--- FEATURE VALUES ---"
)

print(
    "Team 1 Historical Win Rate:",
    round(
        team1_historical * 100,
        2
    ),
    "%"
)

print(
    "Team 2 Historical Win Rate:",
    round(
        team2_historical * 100,
        2
    ),
    "%"
)

print(
    "Team 1 Recent Win Rate:",
    round(
        team1_recent * 100,
        2
    ),
    "%"
)

print(
    "Team 2 Recent Win Rate:",
    round(
        team2_recent * 100,
        2
    ),
    "%"
)

print(
    "Team 1 H2H Win Rate:",
    round(
        team1_h2h * 100,
        2
    ),
    "%"
)

print(
    "Team 2 H2H Win Rate:",
    round(
        team2_h2h * 100,
        2
    ),
    "%"
)

print(
    "Team 1 Venue Win Rate:",
    round(
        team1_venue * 100,
        2
    ),
    "%"
)

print(
    "Team 2 Venue Win Rate:",
    round(
        team2_venue * 100,
        2
    ),
    "%"
)


# -------------------------------------------------
# Prediction Result
# -------------------------------------------------

print(
    "\n--- PREDICTION RESULT ---"
)

if prediction == 1:

    predicted_winner = team1

else:

    predicted_winner = team2


print(
    "Predicted Winner:",
    predicted_winner
)

print(
    team1,
    "Win Probability:",
    round(
        team1_probability * 100,
        2
    ),
    "%"
)

print(
    team2,
    "Win Probability:",
    round(
        team2_probability * 100,
        2
    ),
    "%"
)

print(
    "Prediction Confidence:",
    confidence_level
)


# -------------------------------------------------
# Save Prediction Log
# -------------------------------------------------

log_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\prediction_log.csv"
)

prediction_log = pd.DataFrame([{
    "prediction_time":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

    "team1":
        team1,

    "team2":
        team2,

    "venue":
        venue,

    "toss_winner":
        toss_winner,

    "team1_probability":
        round(
            team1_probability * 100,
            2
        ),

    "team2_probability":
        round(
            team2_probability * 100,
            2
        ),

    "predicted_winner":
        predicted_winner,

    "confidence":
        confidence_level,

    "model_version":
        model_package["model_version"],

    "model_type":
        model_package["model_type"]
}])


# -------------------------------------------------
# Append prediction to existing log
# or create new log if it does not exist
# -------------------------------------------------

if os.path.exists(log_file):

    prediction_log.to_csv(
        log_file,
        mode="a",
        header=False,
        index=False
    )

else:

    prediction_log.to_csv(
        log_file,
        index=False
    )


print(
    "\nPrediction saved to:"
)

print(
    log_file
)