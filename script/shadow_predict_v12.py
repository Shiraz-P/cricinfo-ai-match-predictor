import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from difflib import get_close_matches


ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

MODEL_FILE = (
    ROOT
    / "model"
    / "cricket_model_ablation_9feature.pkl"
)

FEATURE_FILE = (
    ROOT
    / "data"
    / "matches_candidate_features.csv"
)

LOG_FILE = (
    ROOT
    / "data"
    / "shadow_v12_log.csv"
)


FEATURES = [
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
# Common team aliases
# -------------------------------------------------

TEAM_ALIASES = {
    "afgh": "Afghanistan",
    "afg": "Afghanistan",

    "ind": "India",

    "pak": "Pakistan",

    "aus": "Australia",

    "eng": "England",

    "nz": "New Zealand",

    "sa": "South Africa",
    "rsa": "South Africa",

    "sl": "Sri Lanka",

    "wi": "West Indies",

    "ban": "Bangladesh",
    "bd": "Bangladesh",

    "uae": "United Arab Emirates"
}


print("========================================")
print("CANDIDATE v1.2 - SHADOW PREDICTION")
print("========================================")


# -------------------------------------------------
# Load model
# -------------------------------------------------

package = joblib.load(
    MODEL_FILE
)

if isinstance(package, dict):
    model = package["model"]
else:
    model = package


# -------------------------------------------------
# Load historical feature data
# -------------------------------------------------

history = pd.read_csv(
    FEATURE_FILE
)

history["date"] = pd.to_datetime(
    history["date"],
    errors="coerce"
)

history = history.sort_values(
    "date",
    kind="stable"
).reset_index(drop=True)


# -------------------------------------------------
# Known teams and venues
# -------------------------------------------------

known_teams = sorted(
    set(
        history["team1"]
        .dropna()
        .astype(str)
    )
    |
    set(
        history["team2"]
        .dropna()
        .astype(str)
    )
)


known_venues = sorted(
    history["venue"]
    .dropna()
    .astype(str)
    .unique()
)


team_lookup = {
    team.lower(): team
    for team in known_teams
}


venue_lookup = {
    venue.lower(): venue
    for venue in known_venues
}


# -------------------------------------------------
# Normalize team
# -------------------------------------------------

def normalize_team(value):

    value = value.strip()

    if not value:

        raise ValueError(
            "Team name cannot be blank."
        )


    lower = value.lower()


    # Exact case-insensitive match
    if lower in team_lookup:

        return team_lookup[lower]


    # Alias match
    if lower in TEAM_ALIASES:

        canonical = TEAM_ALIASES[
            lower
        ]

        if canonical in known_teams:

            print(
                f"Normalized team: "
                f"{value} -> {canonical}"
            )

            return canonical


    # Partial team match
    partial_matches = [
        team
        for team in known_teams
        if lower in team.lower()
    ]


    if len(partial_matches) == 1:

        resolved = partial_matches[0]

        print(
            f"Resolved team: "
            f"{value} -> {resolved}"
        )

        return resolved


    # Fuzzy suggestions
    suggestions = get_close_matches(
        value,
        known_teams,
        n=5,
        cutoff=0.4
    )


    print(
        f"\nERROR: Unknown team '{value}'."
    )


    if suggestions:

        print(
            "\nDid you mean:"
        )

        for team in suggestions:

            print(
                " -",
                team
            )


    raise ValueError(
        "Unknown team."
    )


# -------------------------------------------------
# Normalize venue
# -------------------------------------------------

def normalize_venue(value):

    value = value.strip()

    if not value:

        raise ValueError(
            "Venue cannot be blank."
        )


    # Explicit unseen venue
    if value.upper() == "NEW":

        print(
            "Using NEW/UNSEEN venue."
        )

        print(
            "Venue features will use "
            "neutral value 0.5."
        )

        return "NEW"


    lower = value.lower()


    # -------------------------------------------------
    # 1. Exact case-insensitive match
    # -------------------------------------------------

    if lower in venue_lookup:

        return venue_lookup[lower]


    # -------------------------------------------------
    # 2. Partial / city match
    #
    # Example:
    # Mumbai -> Wankhede Stadium, Mumbai
    # -------------------------------------------------

    partial_matches = [
        venue
        for venue in known_venues
        if lower in venue.lower()
    ]


    if len(partial_matches) == 1:

        resolved = partial_matches[0]

        print(
            f"Resolved venue: "
            f"{value} -> {resolved}"
        )

        return resolved


    # -------------------------------------------------
    # 3. Multiple stadiums found
    # -------------------------------------------------

    if len(partial_matches) > 1:

        print(
            f"\nMultiple historical venues "
            f"found for '{value}':"
        )

        for i, venue in enumerate(
            partial_matches,
            start=1
        ):

            print(
                f"{i}. {venue}"
            )


        print(
            "\nPlease enter a more specific "
            "stadium or venue name."
        )


        raise ValueError(
            "Venue is ambiguous."
        )


    # -------------------------------------------------
    # 4. Fuzzy matching
    # -------------------------------------------------

    suggestions = get_close_matches(
        value,
        known_venues,
        n=5,
        cutoff=0.35
    )


    print(
        f"\nERROR: Venue '{value}' "
        "was not found."
    )


    if suggestions:

        print(
            "\nClosest known venues:"
        )

        for venue in suggestions:

            print(
                " -",
                venue
            )


    print(
        "\nIf this is genuinely a new venue, "
        "enter NEW."
    )


    raise ValueError(
        "Unknown venue."
    )


# -------------------------------------------------
# Normalize toss winner
# -------------------------------------------------

def normalize_toss(
    value,
    team1,
    team2
):

    value = value.strip()


    # Unknown toss allowed
    if not value:

        return None


    lower = value.lower()


    # Exact Team 1
    if lower == team1.lower():

        return team1


    # Exact Team 2
    if lower == team2.lower():

        return team2


    # Alias check
    if lower in TEAM_ALIASES:

        normalized = TEAM_ALIASES[
            lower
        ]


        if normalized == team1:

            print(
                f"Normalized toss winner: "
                f"{value} -> {team1}"
            )

            return team1


        if normalized == team2:

            print(
                f"Normalized toss winner: "
                f"{value} -> {team2}"
            )

            return team2


    # Partial match against the two teams
    possible_teams = [
        team1,
        team2
    ]


    partial_matches = [
        team
        for team in possible_teams
        if lower in team.lower()
    ]


    if len(partial_matches) == 1:

        resolved = partial_matches[0]

        print(
            f"Resolved toss winner: "
            f"{value} -> {resolved}"
        )

        return resolved


    raise ValueError(
        "Toss winner must match "
        "Team 1 or Team 2, "
        "or be left blank."
    )


# -------------------------------------------------
# User input
# -------------------------------------------------

print(
    "\nEnter match information."
)

print(
    "Team abbreviations such as "
    "Afgh, IND, PAK are supported."
)

print(
    "For venue, you can enter a city "
    "such as Mumbai if it uniquely "
    "matches a historical stadium."
)

print(
    "Use NEW only for a genuinely "
    "unseen venue."
)


raw_team1 = input(
    "\nTeam 1: "
)

raw_team2 = input(
    "Team 2: "
)

raw_venue = input(
    "Venue: "
)

raw_toss = input(
    "Toss winner "
    "(leave blank if unknown): "
)


# -------------------------------------------------
# Validate / normalize
# -------------------------------------------------

try:

    team1 = normalize_team(
        raw_team1
    )

    team2 = normalize_team(
        raw_team2
    )


    if team1 == team2:

        raise ValueError(
            "Team 1 and Team 2 "
            "cannot be the same."
        )


    venue = normalize_venue(
        raw_venue
    )


    toss_winner = normalize_toss(
        raw_toss,
        team1,
        team2
    )


except ValueError as error:

    print("\n========================================")
    print("INPUT VALIDATION FAILED")
    print("========================================")

    print(
        error
    )

    print(
        "\nPrediction NOT generated."
    )

    print(
        "Prediction NOT logged."
    )

    raise SystemExit(1)


# -------------------------------------------------
# Historical helper functions
# -------------------------------------------------

def team_matches(team):

    return history[
        (
            history["team1"]
            ==
            team
        )
        |
        (
            history["team2"]
            ==
            team
        )
    ]


def historical_win_rate(team):

    matches = team_matches(
        team
    )


    if len(matches) == 0:

        return 0.5


    wins = (
        matches["winner"]
        ==
        team
    ).sum()


    return (
        wins
        /
        len(matches)
    )


def recent_win_rate(
    team,
    number=5
):

    matches = (
        team_matches(team)
        .tail(number)
    )


    if len(matches) == 0:

        return 0.5


    wins = (
        matches["winner"]
        ==
        team
    ).sum()


    return (
        wins
        /
        len(matches)
    )


def h2h_win_rate(
    team_a,
    team_b
):

    matches = history[
        (
            (
                history["team1"]
                ==
                team_a
            )
            &
            (
                history["team2"]
                ==
                team_b
            )
        )
        |
        (
            (
                history["team1"]
                ==
                team_b
            )
            &
            (
                history["team2"]
                ==
                team_a
            )
        )
    ]


    if len(matches) == 0:

        return 0.5


    wins = (
        matches["winner"]
        ==
        team_a
    ).sum()


    return (
        wins
        /
        len(matches)
    )


def venue_win_rate(
    team,
    venue_name
):

    if venue_name == "NEW":

        return 0.5


    matches = history[
        (
            history["venue"]
            ==
            venue_name
        )
        &
        (
            (
                history["team1"]
                ==
                team
            )
            |
            (
                history["team2"]
                ==
                team
            )
        )
    ]


    if len(matches) == 0:

        return 0.5


    wins = (
        matches["winner"]
        ==
        team
    ).sum()


    return (
        wins
        /
        len(matches)
    )


# -------------------------------------------------
# Toss feature
# -------------------------------------------------

if toss_winner is None:

    toss_feature = 0.5

elif toss_winner == team1:

    toss_feature = 1.0

else:

    toss_feature = 0.0


# -------------------------------------------------
# Build model features
# -------------------------------------------------

feature_values = {

    "toss_winner_is_team1":
        toss_feature,

    "team1_historical_win_rate":
        historical_win_rate(
            team1
        ),

    "team2_historical_win_rate":
        historical_win_rate(
            team2
        ),

    "team1_recent_win_rate":
        recent_win_rate(
            team1
        ),

    "team2_recent_win_rate":
        recent_win_rate(
            team2
        ),

    "team1_h2h_win_rate":
        h2h_win_rate(
            team1,
            team2
        ),

    "team2_h2h_win_rate":
        h2h_win_rate(
            team2,
            team1
        ),

    "team1_venue_win_rate":
        venue_win_rate(
            team1,
            venue
        ),

    "team2_venue_win_rate":
        venue_win_rate(
            team2,
            venue
        )
}


X = pd.DataFrame(
    [feature_values],
    columns=FEATURES
)


# -------------------------------------------------
# Show validated input
# -------------------------------------------------

print("\n========================================")
print("VALIDATED INPUT")
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
    "Venue:",
    venue
)

print(
    "Toss winner:",
    toss_winner
    if toss_winner
    else "UNKNOWN"
)


# -------------------------------------------------
# Show feature values
# -------------------------------------------------

print("\n========================================")
print("MODEL FEATURES")
print("========================================")

for feature in FEATURES:

    print(
        f"{feature}: "
        f"{feature_values[feature]:.4f}"
    )


# -------------------------------------------------
# Prediction
# -------------------------------------------------

prediction = int(
    model.predict(X)[0]
)


probabilities = (
    model.predict_proba(X)[0]
)


if prediction == 1:

    predicted_winner = team1

else:

    predicted_winner = team2


team1_probability = float(
    probabilities[1]
)

team2_probability = float(
    probabilities[0]
)


# -------------------------------------------------
# Show prediction
# -------------------------------------------------

print("\n========================================")
print("SHADOW RESULT")
print("========================================")

print(
    "Candidate:",
    "v1.2"
)

print(
    "Predicted winner:",
    predicted_winner
)

print(
    f"{team1} probability:",
    f"{team1_probability * 100:.2f}%"
)

print(
    f"{team2} probability:",
    f"{team2_probability * 100:.2f}%"
)


# -------------------------------------------------
# Log valid prediction only
# -------------------------------------------------

row = pd.DataFrame([{

    "prediction_timestamp":
        datetime.now().isoformat(
            timespec="seconds"
        ),

    "team1":
        team1,

    "team2":
        team2,

    "venue":
        venue,

    "toss_winner":
        (
            toss_winner
            if toss_winner
            else "UNKNOWN"
        ),

    "model_version":
        "candidate_v1.2",

    "predicted_winner":
        predicted_winner,

    "team1_probability":
        team1_probability,

    "team2_probability":
        team2_probability,

    "actual_winner":
        "",

    "correct_prediction":
        "",

    "result_status":
        "PENDING"
}])


if LOG_FILE.exists():

    old = pd.read_csv(
        LOG_FILE
    )


    log = pd.concat(
        [
            old,
            row
        ],
        ignore_index=True
    )

else:

    log = row


log.to_csv(
    LOG_FILE,
    index=False
)


print("\n========================================")
print("SHADOW LOG")
print("========================================")

print(
    "Valid prediction logged."
)

print(
    "File:",
    LOG_FILE
)


print(
    "\nChampion v1.0 remains unchanged."
)