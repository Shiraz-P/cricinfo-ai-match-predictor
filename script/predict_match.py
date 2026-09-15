import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from difflib import get_close_matches


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

CHAMPION_MODEL_FILE = (
    MODEL_DIR
    / "cricket_model.pkl"
)

CANDIDATE_MODEL_FILE = (
    MODEL_DIR
    / "cricket_model_ablation_9feature.pkl"
)

CHAMPION_FEATURE_FILE = (
    DATA_DIR
    / "matches_features.csv"
)

CANDIDATE_FEATURE_FILE = (
    DATA_DIR
    / "matches_candidate_features.csv"
)

PRODUCTION_LOG_FILE = (
    DATA_DIR
    / "prediction_log.csv"
)

SHADOW_LOG_FILE = (
    DATA_DIR
    / "shadow_v12_log.csv"
)


# ============================================================
# MODEL FEATURES
# ============================================================

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


# ============================================================
# TEAM ALIASES
# ============================================================

TEAM_ALIASES = {

    "afg": "Afghanistan",
    "afgh": "Afghanistan",

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


# ============================================================
# LOAD MODEL HELPER
# ============================================================

def load_model_package(path):

    package = joblib.load(path)

    if isinstance(package, dict):

        if "model" in package:

            return package["model"]

    return package


# ============================================================
# LOAD MODELS
# ============================================================

champion_model = load_model_package(
    CHAMPION_MODEL_FILE
)

candidate_model = load_model_package(
    CANDIDATE_MODEL_FILE
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

champion_history = pd.read_csv(
    CHAMPION_FEATURE_FILE
)

candidate_history = pd.read_csv(
    CANDIDATE_FEATURE_FILE
)


for df in [
    champion_history,
    candidate_history
]:

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df.sort_values(
        "date",
        inplace=True,
        kind="stable"
    )

    df.reset_index(
        drop=True,
        inplace=True
    )


# ============================================================
# BUILD TEAM LISTS
# ============================================================

champion_teams = sorted(

    set(
        champion_history[
            "team1"
        ]
        .dropna()
        .astype(str)
    )

    |

    set(
        champion_history[
            "team2"
        ]
        .dropna()
        .astype(str)
    )
)


candidate_teams = sorted(

    set(
        candidate_history[
            "team1"
        ]
        .dropna()
        .astype(str)
    )

    |

    set(
        candidate_history[
            "team2"
        ]
        .dropna()
        .astype(str)
    )
)


candidate_team_lookup = {

    team.lower(): team

    for team in candidate_teams
}


# ============================================================
# NORMALIZE TEAM
# ============================================================

def normalize_team(value):

    value = value.strip()

    if not value:

        raise ValueError(
            "Team name cannot be blank."
        )


    lower = value.lower()


    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    if lower in candidate_team_lookup:

        return candidate_team_lookup[
            lower
        ]


    # --------------------------------------------------------
    # Alias match
    # --------------------------------------------------------

    if lower in TEAM_ALIASES:

        canonical = TEAM_ALIASES[
            lower
        ]

        if canonical in candidate_teams:

            print(
                f"Normalized team: "
                f"{value} -> {canonical}"
            )

            return canonical


    # --------------------------------------------------------
    # Partial match
    # --------------------------------------------------------

    partial_matches = [

        team

        for team in candidate_teams

        if lower in team.lower()
    ]


    if len(partial_matches) == 1:

        resolved = partial_matches[0]

        print(
            f"Resolved team: "
            f"{value} -> {resolved}"
        )

        return resolved


    # --------------------------------------------------------
    # Fuzzy suggestions
    # --------------------------------------------------------

    suggestions = get_close_matches(
        value,
        candidate_teams,
        n=5,
        cutoff=0.4
    )


    print(
        f"\nERROR: Team '{value}' "
        "was not found."
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


# ============================================================
# SELECT MODEL AUTOMATICALLY
# ============================================================

def select_model(
    team1,
    team2
):

    # --------------------------------------------------------
    # Both teams supported by Champion
    # --------------------------------------------------------

    if (
        team1 in champion_teams
        and
        team2 in champion_teams
    ):

        return (
            champion_model,
            champion_history,
            "Champion v1.0",
            "PRODUCTION"
        )


    # --------------------------------------------------------
    # Otherwise try expanded Candidate
    # --------------------------------------------------------

    if (
        team1 in candidate_teams
        and
        team2 in candidate_teams
    ):

        return (
            candidate_model,
            candidate_history,
            "Candidate v1.2",
            "SHADOW"
        )


    raise ValueError(
        "This team combination is not "
        "supported by the available models."
    )


# ============================================================
# NORMALIZE VENUE
# ============================================================

def normalize_venue(
    value,
    history
):

    value = value.strip()

    if not value:

        raise ValueError(
            "Venue cannot be blank."
        )


    # --------------------------------------------------------
    # Explicit unseen venue
    # --------------------------------------------------------

    if value.upper() == "NEW":

        print(
            "Using NEW/UNSEEN venue."
        )

        print(
            "Neutral venue features "
            "will be used."
        )

        return "NEW"


    # --------------------------------------------------------
    # Known venues from selected model history
    # --------------------------------------------------------

    known_venues = sorted(

        history[
            "venue"
        ]
        .dropna()
        .astype(str)
        .unique()
    )


    venue_lookup = {

        venue.lower(): venue

        for venue in known_venues
    }


    lower = value.lower()


    # --------------------------------------------------------
    # 1. Exact match
    # --------------------------------------------------------

    if lower in venue_lookup:

        return venue_lookup[
            lower
        ]


    # --------------------------------------------------------
    # 2. Partial / city match
    #
    # Examples:
    #
    # Mumbai
    #   -> Wankhede Stadium, Mumbai
    #
    # Dubai
    #   -> most frequently observed Dubai venue
    # --------------------------------------------------------

    partial_matches = [

        venue

        for venue in known_venues

        if lower in venue.lower()
    ]


    # --------------------------------------------------------
    # One matching venue
    # --------------------------------------------------------

    if len(partial_matches) == 1:

        resolved = partial_matches[0]

        print(
            f"Resolved venue: "
            f"{value} -> {resolved}"
        )

        return resolved


    # --------------------------------------------------------
    # Multiple venues for same city/input
    #
    # Automatically select most frequently observed
    # historical venue.
    # --------------------------------------------------------

    if len(partial_matches) > 1:

        venue_counts = (

            history[
                history["venue"]
                .isin(partial_matches)
            ]["venue"]
            .value_counts()
        )


        resolved = venue_counts.index[0]


        print(
            f"Resolved city: "
            f"{value} -> {resolved}"
        )


        print(
            "Reason: most frequently observed "
            "historical venue for this input."
        )


        print(
            "Other matching venues:"
        )


        for venue in partial_matches:

            if venue != resolved:

                print(
                    " -",
                    venue
                )


        return resolved


    # --------------------------------------------------------
    # 3. Fuzzy matching
    # --------------------------------------------------------

    suggestions = get_close_matches(
        value,
        known_venues,
        n=5,
        cutoff=0.35
    )


    # --------------------------------------------------------
    # Exactly one fuzzy suggestion
    # --------------------------------------------------------

    if len(suggestions) == 1:

        resolved = suggestions[0]

        print(
            f"Resolved venue: "
            f"{value} -> {resolved}"
        )

        return resolved


    # --------------------------------------------------------
    # Unknown venue
    # --------------------------------------------------------

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


# ============================================================
# NORMALIZE TOSS WINNER
# ============================================================

def normalize_toss(
    value,
    team1,
    team2
):

    value = value.strip()


    # --------------------------------------------------------
    # Unknown toss allowed
    # --------------------------------------------------------

    if not value:

        return None


    lower = value.lower()


    # --------------------------------------------------------
    # Exact Team 1
    # --------------------------------------------------------

    if lower == team1.lower():

        return team1


    # --------------------------------------------------------
    # Exact Team 2
    # --------------------------------------------------------

    if lower == team2.lower():

        return team2


    # --------------------------------------------------------
    # Alias
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Partial team name
    # --------------------------------------------------------

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


# ============================================================
# HISTORICAL FEATURE HELPERS
# ============================================================

def team_matches(
    history,
    team
):

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


# ============================================================
# HISTORICAL WIN RATE
# ============================================================

def historical_win_rate(
    history,
    team
):

    matches = team_matches(
        history,
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


# ============================================================
# RECENT WIN RATE
# ============================================================

def recent_win_rate(
    history,
    team,
    number=5
):

    matches = (
        team_matches(
            history,
            team
        )
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


# ============================================================
# HEAD-TO-HEAD WIN RATE
# ============================================================

def h2h_win_rate(
    history,
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


# ============================================================
# VENUE WIN RATE
# ============================================================

def venue_win_rate(
    history,
    team,
    venue
):

    if venue == "NEW":

        return 0.5


    matches = history[

        (
            history["venue"]
            ==
            venue
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


# ============================================================
# APPEND PREDICTION TO LOG
# ============================================================

def append_log(
    file_path,
    row
):

    new_row = pd.DataFrame(
        [row]
    )


    if file_path.exists():

        old = pd.read_csv(
            file_path
        )

        final = pd.concat(
            [
                old,
                new_row
            ],
            ignore_index=True
        )

    else:

        final = new_row


    final.to_csv(
        file_path,
        index=False
    )


# ============================================================
# START PREDICTOR
# ============================================================

print("\n========================================")
print("T20 CRICKET AI PREDICTOR")
print("========================================")

print(
    "\nOne prediction interface."
)

print(
    "The system automatically selects "
    "the appropriate model."
)


# ============================================================
# GET TEAMS
# ============================================================

raw_team1 = input(
    "\nTeam 1: "
)

raw_team2 = input(
    "Team 2: "
)


# ============================================================
# VALIDATE TEAMS AND SELECT MODEL
# ============================================================

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


    (
        model,
        history,
        model_name,
        model_mode
    ) = select_model(
        team1,
        team2
    )


    print("\n----------------------------------------")
    print("MODEL ROUTING")
    print("----------------------------------------")

    print(
        "Selected model:",
        model_name
    )

    print(
        "Mode:",
        model_mode
    )


    if model_mode == "SHADOW":

        print(
            "\nNOTE:"
        )

        print(
            "Champion v1.0 does not support "
            "this complete team combination."
        )

        print(
            "Candidate v1.2 is being used "
            "in SHADOW mode."
        )


    # ========================================================
    # GET VENUE
    # ========================================================

    raw_venue = input(
        "\nVenue: "
    )


    venue = normalize_venue(
        raw_venue,
        history
    )


    # ========================================================
    # GET TOSS
    # ========================================================

    raw_toss = input(
        "Toss winner "
        "(leave blank if unknown): "
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

    print(error)

    print(
        "\nPrediction NOT generated."
    )

    print(
        "Prediction NOT logged."
    )

    raise SystemExit(1)


# ============================================================
# TOSS FEATURE
# ============================================================

if toss_winner is None:

    toss_feature = 0.5


elif toss_winner == team1:

    toss_feature = 1.0


else:

    toss_feature = 0.0


# ============================================================
# BUILD FEATURE VALUES
# ============================================================

feature_values = {

    "toss_winner_is_team1":
        toss_feature,

    "team1_historical_win_rate":
        historical_win_rate(
            history,
            team1
        ),

    "team2_historical_win_rate":
        historical_win_rate(
            history,
            team2
        ),

    "team1_recent_win_rate":
        recent_win_rate(
            history,
            team1
        ),

    "team2_recent_win_rate":
        recent_win_rate(
            history,
            team2
        ),

    "team1_h2h_win_rate":
        h2h_win_rate(
            history,
            team1,
            team2
        ),

    "team2_h2h_win_rate":
        h2h_win_rate(
            history,
            team2,
            team1
        ),

    "team1_venue_win_rate":
        venue_win_rate(
            history,
            team1,
            venue
        ),

    "team2_venue_win_rate":
        venue_win_rate(
            history,
            team2,
            venue
        )
}


# ============================================================
# CREATE MODEL INPUT
# ============================================================

X = pd.DataFrame(
    [feature_values],
    columns=FEATURES
)


# ============================================================
# DISPLAY VALIDATED INPUT
# ============================================================

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


# ============================================================
# DISPLAY MODEL FEATURES
# ============================================================

print("\n========================================")
print("MODEL FEATURES")
print("========================================")


for feature in FEATURES:

    print(
        f"{feature}: "
        f"{feature_values[feature]:.4f}"
    )


# ============================================================
# GENERATE PREDICTION
# ============================================================

prediction = int(
    model.predict(X)[0]
)


probabilities = (
    model.predict_proba(X)[0]
)


# ============================================================
# DETERMINE WINNER
# ============================================================

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


# ============================================================
# DISPLAY PREDICTION
# ============================================================

print("\n========================================")
print("PREDICTION RESULT")
print("========================================")

print(
    "Model:",
    model_name
)

print(
    "Mode:",
    model_mode
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
    "Toss winner:",
    toss_winner
    if toss_winner
    else "UNKNOWN"
)


print(
    "\nPredicted winner:"
)

print(
    predicted_winner
)


print(
    f"\n{team1}: "
    f"{team1_probability * 100:.2f}%"
)


print(
    f"{team2}: "
    f"{team2_probability * 100:.2f}%"
)


# ============================================================
# PREPARE LOG RECORD
# ============================================================

timestamp = datetime.now().isoformat(
    timespec="seconds"
)


log_row = {

    "prediction_timestamp":
        timestamp,

    "team1":
        team1,

    "team2":
        team2,

    "venue":
        venue,

    "toss_winner":
        toss_winner
        if toss_winner
        else "UNKNOWN",

    "model_version":
        (
            "champion_v1.0"
            if model_mode == "PRODUCTION"
            else "candidate_v1.2"
        ),

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
}


# ============================================================
# LOG ACCORDING TO MODEL MODE
# ============================================================

if model_mode == "PRODUCTION":

    append_log(
        PRODUCTION_LOG_FILE,
        log_row
    )


    print("\n========================================")
    print("PRODUCTION LOG")
    print("========================================")

    print(
        "Champion prediction logged."
    )

    print(
        "File:",
        PRODUCTION_LOG_FILE
    )


else:

    append_log(
        SHADOW_LOG_FILE,
        log_row
    )


    print("\n========================================")
    print("SHADOW LOG")
    print("========================================")

    print(
        "Candidate prediction logged."
    )

    print(
        "File:",
        SHADOW_LOG_FILE
    )


# ============================================================
# FINAL MODEL GOVERNANCE MESSAGE
# ============================================================

print("\n========================================")

if model_mode == "PRODUCTION":

    print(
        "Champion v1.0 prediction completed."
    )

    print(
        "Mode: PRODUCTION"
    )


else:

    print(
        "Candidate v1.2 shadow prediction completed."
    )

    print(
        "Mode: SHADOW"
    )

    print(
        "Champion v1.0 remains production."
    )


print("========================================")