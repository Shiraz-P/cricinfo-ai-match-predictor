from difflib import get_close_matches
from datetime import datetime

import os
import joblib
import pandas as pd
import numpy as np


# -------------------------------------------------
# Load feature-engineered dataset
# -------------------------------------------------

feature_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\matches_features.csv"
)

reliability_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\feature_reliability_analysis.csv"
)

df = pd.read_csv(
    feature_file
)

reliability_df = pd.read_csv(
    reliability_file
)


df["date"] = pd.to_datetime(
    df["date"]
)

reliability_df["date"] = pd.to_datetime(
    reliability_df["date"]
)


df = df.sort_values(
    "date"
).reset_index(
    drop=True
)

reliability_df = reliability_df.sort_values(
    "date"
).reset_index(
    drop=True
)


# -------------------------------------------------
# Safety Check
# -------------------------------------------------

if len(df) != len(reliability_df):

    raise ValueError(
        "Feature dataset and reliability dataset "
        "do not contain the same number of rows."
    )


# -------------------------------------------------
# Load Champion v1.0
# -------------------------------------------------

champion_path = (
    r"K:\Python\Cricinfo_AI_Project\model"
    r"\cricket_model.pkl"
)

champion_package = joblib.load(
    champion_path
)

champion_model = champion_package[
    "model"
]

champion_features = champion_package[
    "features"
]


# -------------------------------------------------
# Load Challenger v1.1
# -------------------------------------------------

challenger_path = (
    r"K:\Python\Cricinfo_AI_Project\model"
    r"\cricket_model_v1_1_candidate.pkl"
)

challenger_package = joblib.load(
    challenger_path
)

challenger_model = challenger_package[
    "model"
]

challenger_features = challenger_package[
    "features"
]


# -------------------------------------------------
# Show Model Information
# -------------------------------------------------

print("\n================================")
print("CHAMPION VS CHALLENGER")
print("================================")

print(
    "Champion version:",
    champion_package[
        "model_version"
    ]
)

print(
    "Champion features:",
    len(
        champion_features
    )
)

print(
    "Challenger version:",
    challenger_package[
        "model_version"
    ]
)

print(
    "Challenger features:",
    len(
        challenger_features
    )
)


# -------------------------------------------------
# Build Official Team List
# -------------------------------------------------

all_teams = sorted(
    set(
        df["team1"]
    ).union(
        set(
            df["team2"]
        )
    )
)


# -------------------------------------------------
# Function: Normalize Team Name
# -------------------------------------------------

def normalize_team(user_input):

    normalized_input = (
        user_input
        .strip()
        .lower()
        .replace(" ", "")
    )


    # ---------------------------------------------
    # Exact normalized match
    # ---------------------------------------------

    for team in all_teams:

        normalized_team = (
            team
            .lower()
            .replace(" ", "")
        )

        if normalized_team == normalized_input:

            return team


    # ---------------------------------------------
    # Fuzzy matching
    # ---------------------------------------------

    normalized_team_map = {

        team
        .lower()
        .replace(" ", ""):
        team

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
# Function: Normalize Generic Text
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
# Function: Resolve Venue
#
# Order:
# 1. Exact normalized match
# 2. Partial match
# 3. Fuzzy match
# -------------------------------------------------

def resolve_venue(user_input):

    normalized_input = normalize_text(
        user_input
    )

    all_venues = sorted(
        df[
            "venue"
        ]
        .dropna()
        .unique()
    )


    # ---------------------------------------------
    # Exact normalized match
    # Example:
    # lords -> Lord's
    # ---------------------------------------------

    for venue in all_venues:

        if (
            normalize_text(
                venue
            )
            ==
            normalized_input
        ):

            return venue


    # ---------------------------------------------
    # Partial match
    # Example:
    # sydney -> Sydney Cricket Ground
    # ---------------------------------------------

    matches = [

        venue

        for venue in all_venues

        if normalized_input
        in normalize_text(
            venue
        )
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

                choice_number = int(
                    choice
                )

                if (
                    1
                    <=
                    choice_number
                    <=
                    len(matches)
                ):

                    return matches[
                        choice_number - 1
                    ]


            print(
                "Invalid choice. "
                "Please enter one of the "
                "venue numbers shown."
            )


    # ---------------------------------------------
    # Fuzzy venue matching
    # ---------------------------------------------

    normalized_venue_map = {

        normalize_text(
            venue
        ):
        venue

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

            normalized_venue_map[
                item
            ]

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

                choice_number = int(
                    choice
                )

                if (
                    1
                    <=
                    choice_number
                    <=
                    len(
                        suggested_venues
                    )
                ):

                    return suggested_venues[
                        choice_number - 1
                    ]


            print(
                "Invalid choice."
            )


    return None


# -------------------------------------------------
# Function: Team Historical Win Rate
# -------------------------------------------------

def historical_win_rate(team):

    matches = df[
        (
            df["team1"]
            ==
            team
        )
        |
        (
            df["team2"]
            ==
            team
        )
    ]


    if len(matches) == 0:

        return 0.5


    wins = len(
        matches[
            matches[
                "winner"
            ]
            ==
            team
        ]
    )


    return (
        wins
        /
        len(matches)
    )


# -------------------------------------------------
# Function: Recent Win Rate
# -------------------------------------------------

def recent_win_rate(team):

    matches = df[
        (
            df["team1"]
            ==
            team
        )
        |
        (
            df["team2"]
            ==
            team
        )
    ].sort_values(
        "date"
    )


    last_five = matches.tail(
        5
    )


    if len(last_five) == 0:

        return 0.5


    wins = len(
        last_five[
            last_five[
                "winner"
            ]
            ==
            team
        ]
    )


    return (
        wins
        /
        len(last_five)
    )


# -------------------------------------------------
# Function: Head-to-Head Win Rate
# -------------------------------------------------

def head_to_head(
    team1,
    team2
):

    matches = df[
        (
            (
                df["team1"]
                ==
                team1
            )
            &
            (
                df["team2"]
                ==
                team2
            )
        )
        |
        (
            (
                df["team1"]
                ==
                team2
            )
            &
            (
                df["team2"]
                ==
                team1
            )
        )
    ]


    if len(matches) == 0:

        return (
            0.5,
            0.5
        )


    total = len(
        matches
    )


    team1_wins = len(
        matches[
            matches[
                "winner"
            ]
            ==
            team1
        ]
    )


    team2_wins = len(
        matches[
            matches[
                "winner"
            ]
            ==
            team2
        ]
    )


    return (
        team1_wins
        /
        total,

        team2_wins
        /
        total
    )


# -------------------------------------------------
# Function: Venue Win Rate
# -------------------------------------------------

def venue_win_rate(
    team,
    venue
):

    matches = df[
        (
            (
                df["team1"]
                ==
                team
            )
            |
            (
                df["team2"]
                ==
                team
            )
        )
        &
        (
            df["venue"]
            ==
            venue
        )
    ]


    if len(matches) == 0:

        return 0.5


    wins = len(
        matches[
            matches[
                "winner"
            ]
            ==
            team
        ]
    )


    return (
        wins
        /
        len(matches)
    )


# -------------------------------------------------
# Function: Historical Sample Size
# -------------------------------------------------

def historical_sample_size(
    team
):

    matches = df[
        (
            df["team1"]
            ==
            team
        )
        |
        (
            df["team2"]
            ==
            team
        )
    ]


    return len(
        matches
    )


# =================================================
# USER INPUT
# =================================================


# -------------------------------------------------
# Team 1 Input Validation
# -------------------------------------------------

while True:

    team1_input = input(
        "\nEnter Team 1: "
    )


    team1 = normalize_team(
        team1_input
    )


    if team1 is not None:

        break


    print(
        "Team not found. "
        "Please try again."
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
            "Team not found. "
            "Please try again."
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
        "Please try again."
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
            "Toss winner must be either",
            team1,
            "or",
            team2
        )

        continue


    break


# -------------------------------------------------
# Show Resolved Inputs
# -------------------------------------------------

print(
    "\n--- RESOLVED MATCH INPUT ---"
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


# =================================================
# FEATURE CALCULATION
# =================================================


# -------------------------------------------------
# Toss Feature
# -------------------------------------------------

toss_winner_is_team1 = (
    1
    if toss_winner == team1
    else 0
)


# -------------------------------------------------
# Historical Win Rates
# -------------------------------------------------

team1_historical = historical_win_rate(
    team1
)

team2_historical = historical_win_rate(
    team2
)


# -------------------------------------------------
# Recent Form
# -------------------------------------------------

team1_recent = recent_win_rate(
    team1
)

team2_recent = recent_win_rate(
    team2
)


# -------------------------------------------------
# Head-to-Head
# -------------------------------------------------

team1_h2h, team2_h2h = head_to_head(
    team1,
    team2
)


# -------------------------------------------------
# Venue Win Rates
# -------------------------------------------------

team1_venue = venue_win_rate(
    team1,
    venue
)

team2_venue = venue_win_rate(
    team2,
    venue
)


# -------------------------------------------------
# Champion Input
# -------------------------------------------------

champion_input = pd.DataFrame([{

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
# Challenger Historical Sample Reliability
# -------------------------------------------------

team1_sample = historical_sample_size(
    team1
)

team2_sample = historical_sample_size(
    team2
)


team1_sample_log = np.log1p(
    team1_sample
)

team2_sample_log = np.log1p(
    team2_sample
)


# -------------------------------------------------
# Challenger Input
# -------------------------------------------------

challenger_input = champion_input.copy()


challenger_input[
    "team1_historical_sample_log"
] = team1_sample_log


challenger_input[
    "team2_historical_sample_log"
] = team2_sample_log


# -------------------------------------------------
# Feature Schema Safety Checks
# -------------------------------------------------

champion_input = champion_input[
    champion_features
]


challenger_input = challenger_input[
    challenger_features
]


# =================================================
# MODEL PREDICTIONS
# =================================================


# -------------------------------------------------
# Champion Prediction
# -------------------------------------------------

champion_prediction = champion_model.predict(
    champion_input
)[0]


champion_probabilities = (
    champion_model.predict_proba(
        champion_input
    )[0]
)


# -------------------------------------------------
# Challenger Prediction
# -------------------------------------------------

challenger_prediction = challenger_model.predict(
    challenger_input
)[0]


challenger_probabilities = (
    challenger_model.predict_proba(
        challenger_input
    )[0]
)


# -------------------------------------------------
# Convert Predictions to Team Names
# -------------------------------------------------

champion_winner = (
    team1
    if champion_prediction == 1
    else team2
)


challenger_winner = (
    team1
    if challenger_prediction == 1
    else team2
)


# =================================================
# OUTPUT
# =================================================


# -------------------------------------------------
# Champion Result
# -------------------------------------------------

print(
    "\n--- CHAMPION V1.0 ---"
)

print(
    "Predicted Winner:",
    champion_winner
)

print(
    team1,
    "Probability:",
    round(
        champion_probabilities[1]
        * 100,
        2
    ),
    "%"
)

print(
    team2,
    "Probability:",
    round(
        champion_probabilities[0]
        * 100,
        2
    ),
    "%"
)


# -------------------------------------------------
# Challenger Result
# -------------------------------------------------

print(
    "\n--- CHALLENGER V1.1 ---"
)

print(
    "Predicted Winner:",
    challenger_winner
)

print(
    team1,
    "Probability:",
    round(
        challenger_probabilities[1]
        * 100,
        2
    ),
    "%"
)

print(
    team2,
    "Probability:",
    round(
        challenger_probabilities[0]
        * 100,
        2
    ),
    "%"
)


# -------------------------------------------------
# Champion vs Challenger Comparison
# -------------------------------------------------

print(
    "\n--- COMPARISON ---"
)


if (
    champion_winner
    ==
    challenger_winner
):

    print(
        "Both models predict the same winner."
    )

else:

    print(
        "Models disagree on the predicted winner."
    )


# -------------------------------------------------
# Probability Difference
# -------------------------------------------------

probability_difference = abs(
    challenger_probabilities[1]
    -
    champion_probabilities[1]
)


print(
    "Team 1 probability difference:",
    round(
        probability_difference
        * 100,
        2
    ),
    "percentage points"
)


print(
    "\nChampion:",
    champion_package[
        "model_version"
    ]
)


print(
    "Challenger:",
    challenger_package[
        "model_version"
    ]
)


print(
    "\nProduction model has not been changed."
)


# =================================================
# SHADOW COMPARISON LOGGING
# =================================================


# -------------------------------------------------
# Shadow Log File
# -------------------------------------------------

shadow_log_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\shadow_comparison_log.csv"
)


# -------------------------------------------------
# Model Agreement
# -------------------------------------------------

models_agree = (
    champion_winner
    ==
    challenger_winner
)


# -------------------------------------------------
# Probability Difference in Percentage Points
# -------------------------------------------------

probability_difference_pp = round(
    probability_difference
    * 100,
    2
)


# -------------------------------------------------
# Confidence Levels
# -------------------------------------------------

champion_confidence = max(
    champion_probabilities
)


challenger_confidence = max(
    challenger_probabilities
)


def get_confidence_level(
    confidence
):

    if confidence >= 0.70:

        return "HIGH"

    elif confidence >= 0.60:

        return "MEDIUM"

    else:

        return "LOW"


champion_confidence_level = get_confidence_level(
    champion_confidence
)


challenger_confidence_level = get_confidence_level(
    challenger_confidence
)


# -------------------------------------------------
# Build Shadow Record
# -------------------------------------------------

shadow_record = pd.DataFrame([{

    # ---------------------------------------------
    # Match Information
    # ---------------------------------------------

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


    # ---------------------------------------------
    # Champion Information
    # ---------------------------------------------

    "champion_version":
        champion_package[
            "model_version"
        ],

    "champion_winner":
        champion_winner,

    "champion_team1_probability":
        round(
            champion_probabilities[1]
            * 100,
            2
        ),

    "champion_team2_probability":
        round(
            champion_probabilities[0]
            * 100,
            2
        ),

    "champion_confidence":
        round(
            champion_confidence
            * 100,
            2
        ),

    "champion_confidence_level":
        champion_confidence_level,


    # ---------------------------------------------
    # Challenger Information
    # ---------------------------------------------

    "challenger_version":
        challenger_package[
            "model_version"
        ],

    "challenger_winner":
        challenger_winner,

    "challenger_team1_probability":
        round(
            challenger_probabilities[1]
            * 100,
            2
        ),

    "challenger_team2_probability":
        round(
            challenger_probabilities[0]
            * 100,
            2
        ),

    "challenger_confidence":
        round(
            challenger_confidence
            * 100,
            2
        ),

    "challenger_confidence_level":
        challenger_confidence_level,


    # ---------------------------------------------
    # Comparison Information
    # ---------------------------------------------

    "models_agree":
        models_agree,

    "probability_difference_pp":
        probability_difference_pp,


    # ---------------------------------------------
    # Ground Truth
    # Filled Later
    # ---------------------------------------------

    "actual_winner":
        "",

    "champion_correct":
        "",

    "challenger_correct":
        "",

    "result_status":
        "PENDING"

}])


# -------------------------------------------------
# Save / Append Shadow Log
# -------------------------------------------------

try:

    if os.path.exists(
        shadow_log_file
    ):

        shadow_record.to_csv(
            shadow_log_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        shadow_record.to_csv(
            shadow_log_file,
            index=False
        )


    print(
        "\n--- SHADOW COMPARISON LOGGED ---"
    )

    print(
        "Shadow comparison saved to:"
    )

    print(
        shadow_log_file
    )

    print(
        "Champion confidence:",
        champion_confidence_level
    )

    print(
        "Challenger confidence:",
        challenger_confidence_level
    )

    print(
        "Result Status: PENDING"
    )


except PermissionError:

    print(
        "\nERROR:"
    )

    print(
        "Could not write shadow comparison log."
    )

    print(
        "Please close shadow_comparison_log.csv "
        "if it is open in another application."
    )