import re
import time
import requests
import pandas as pd

from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin


PROJECT_ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "matches_afghanistan_verified.csv"
)

YEARS = [2023, 2024, 2025, 2026]

BASE_URL = "https://cricstats.in"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


# -------------------------------------------------
# Team aliases
# -------------------------------------------------

TEAM_ALIASES = {
    "AFG": "Afghanistan",
    "Afghanistan": "Afghanistan",

    "BAN": "Bangladesh",
    "Bangladesh": "Bangladesh",

    "UAE": "United Arab Emirates",
    "U.A.E.": "United Arab Emirates",
    "United Arab Emirates": "United Arab Emirates",

    "IND": "India",
    "India": "India",

    "PAK": "Pakistan",
    "Pakistan": "Pakistan",

    "SL": "Sri Lanka",
    "Sri Lanka": "Sri Lanka",

    "WI": "West Indies",
    "West Indies": "West Indies",

    "NZ": "New Zealand",
    "New Zealand": "New Zealand",

    "SA": "South Africa",
    "South Africa": "South Africa",

    "AUS": "Australia",
    "Australia": "Australia",

    "IRE": "Ireland",
    "Ireland": "Ireland",

    "ZIM": "Zimbabwe",
    "Zimbabwe": "Zimbabwe",

    "UGA": "Uganda",
    "Uganda": "Uganda",

    "PNG": "Papua New Guinea",
    "Papua New Guinea": "Papua New Guinea",

    "HK": "Hong Kong",
    "Hong Kong": "Hong Kong",

    "QTR": "Qatar",
    "Qatar": "Qatar",

    "CAN": "Canada",
    "Canada": "Canada",
}


def clean_text(value):
    return " ".join(str(value).split()).strip()


def get_page(url):

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    return BeautifulSoup(
        response.text,
        "html.parser"
    )


def normalize_team(team):

    team = clean_text(team)

    return TEAM_ALIASES.get(
        team,
        team
    )


def is_afghanistan_a(result_text):

    if not result_text:
        return False

    text = clean_text(result_text).lower()

    suspicious = [
        "afg a won",
        "afghanistan a won",
        "afghan abdalyan won"
    ]

    return any(
        item in text
        for item in suspicious
    )

def extract_winner(result_text):

    result_lower = result_text.lower()

    if (
        "no result" in result_lower
        or
        "abandoned" in result_lower
    ):
        return "No Result"

    # Important aliases
    winner_aliases = {
        "u.a.e. won": "United Arab Emirates",
        "uae won": "United Arab Emirates",

        "afghanistan won": "Afghanistan",
        "bangladesh won": "Bangladesh",
        "india won": "India",
        "pakistan won": "Pakistan",
        "sri lanka won": "Sri Lanka",
        "west indies won": "West Indies",
        "new zealand won": "New Zealand",
        "south africa won": "South Africa",
        "australia won": "Australia",
        "ireland won": "Ireland",
        "zimbabwe won": "Zimbabwe",
        "uganda won": "Uganda",
        "papua new guinea won": "Papua New Guinea",
        "hong kong won": "Hong Kong",
        "qatar won": "Qatar",
        "canada won": "Canada",
    }

    for phrase, team in winner_aliases.items():

        if phrase in result_lower:
            return team

    return None


# -------------------------------------------------
# STEP 1
# Collect Afghanistan candidate match links
# -------------------------------------------------

match_links = set()

for year in YEARS:

    url = f"{BASE_URL}/t20i/results/{year}"

    print("\nScanning:", url)

    soup = get_page(url)

    for link in soup.find_all(
        "a",
        href=True
    ):

        href = link["href"]

        if "/live-cricket-score/" not in href:
            continue

        text = clean_text(
            link.get_text(
                " ",
                strip=True
            )
        )

        if "Afghanistan" not in text:
            continue

        full_url = urljoin(
            BASE_URL,
            href
        )

        match_links.add(
            full_url
        )


print(
    "\nCandidate Afghanistan match links:",
    len(match_links)
)


# -------------------------------------------------
# STEP 2
# Visit individual match pages
# -------------------------------------------------

rows = []

for number, match_url in enumerate(
    sorted(match_links),
    start=1
):

    print(
        f"\n[{number}/{len(match_links)}]",
        match_url
    )

    try:

        soup = get_page(
            match_url
        )

    except Exception as error:

        print(
            "Skipped: request error:",
            error
        )

        continue

    page_text = clean_text(
        soup.get_text(
            " ",
            strip=True
        )
    )


    # -------------------------------------------------
    # Must contain T20I
    # -------------------------------------------------

    if "T20I" not in page_text:

        print(
            "Skipped: not T20I"
        )

        continue


    # -------------------------------------------------
    # Afghanistan must appear
    # -------------------------------------------------

    if "Afghanistan" not in page_text:

        print(
            "Skipped: Afghanistan missing"
        )

        continue


    # -------------------------------------------------
    # Reject Afghanistan A / Abdalyan
    # -------------------------------------------------

       # -------------------------------------------------
    # Date
    # -------------------------------------------------

    date_match = re.search(
        r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+"
        r"(\d{1,2})\s+"
        r"([A-Za-z]{3})\s+"
        r"(\d{4})",
        page_text
    )

    if not date_match:

        print(
            "Skipped: date not found"
        )

        continue


    match_date = pd.to_datetime(
        (
            f"{date_match.group(2)} "
            f"{date_match.group(3)} "
            f"{date_match.group(4)}"
        ),
        format="%d %b %Y",
        errors="coerce"
    )

    if pd.isna(
        match_date
    ):

        print(
            "Skipped: invalid date"
        )

        continue


    # Historical supplemental file already
    # covers through 27-Mar-2023
    if (
        match_date
        <=
        pd.Timestamp(
            "2023-03-27"
        )
    ):

        print(
            "Skipped: already historical"
        )

        continue


    # -------------------------------------------------
    # Extract text tokens
    # -------------------------------------------------

    text_parts = [
        clean_text(text)
        for text in soup.stripped_strings
    ]


    # -------------------------------------------------
    # Find VS
    # -------------------------------------------------

    vs_index = None

    for index, value in enumerate(
        text_parts
    ):

        if value.upper() == "VS":

            vs_index = index
            break


    if vs_index is None:

        print(
            "Skipped: VS not found"
        )

        continue


    before_vs = text_parts[
        max(
            0,
            vs_index - 10
        ):
        vs_index
    ]

    after_vs = text_parts[
        vs_index + 1:
        vs_index + 11
    ]


    # -------------------------------------------------
    # Team detection using aliases
    # -------------------------------------------------

    team1 = None
    team2 = None


    for value in reversed(
        before_vs
    ):

        normalized = normalize_team(
            value
        )

        if normalized in TEAM_ALIASES.values():

            team1 = normalized
            break


    for value in after_vs:

        normalized = normalize_team(
            value
        )

        if normalized in TEAM_ALIASES.values():

            team2 = normalized
            break


    if (
        team1 is None
        or
        team2 is None
    ):

        print(
            "Skipped: teams unresolved"
        )

        continue


    if (
        "Afghanistan"
        not in [
            team1,
            team2
        ]
    ):

        print(
            "Skipped: Afghanistan not one of teams"
        )

        continue


    # -------------------------------------------------
    # Find result
    # -------------------------------------------------

    result_text = None

    for value in text_parts:

        lower_value = value.lower()

        if (
            " won by " in lower_value
            or
            "super over" in lower_value
            or
            "no result" in lower_value
            or
            "abandoned" in lower_value
        ):

            result_text = value
            break


    if result_text is None:

        print(
            "Skipped: result not found"
        )

        continue


    # -------------------------------------------------
    # Extra safety against AFG A
    # -------------------------------------------------

    if (
        "afg a won" in result_text.lower()
        or
        "afghanistan a won"
        in result_text.lower()
    ):

        print(
            "EXCLUDED: Afghanistan A result"
        )

        continue


    winner = extract_winner(
        result_text
    )


    # -------------------------------------------------
    # Super Over special handling
    # -------------------------------------------------

    if winner is None:

        lower_result = result_text.lower()

        for alias, team in TEAM_ALIASES.items():

            if (
                alias.lower()
                in lower_result
                and
                "super over"
                in lower_result
            ):

                winner = team
                break


    if winner is None:

        print(
            "Skipped: winner unresolved:",
            result_text
        )

        continue


    # Winner must be one of the teams
    # except No Result
    if (
        winner != "No Result"
        and
        winner not in [
            team1,
            team2
        ]
    ):

        print(
            "EXCLUDED: winner does not match teams:",
            winner
        )

        continue


    # -------------------------------------------------
    # Venue
    # -------------------------------------------------

    venue = "Unknown"

    date_position = page_text.find(
        date_match.group(0)
    )

    if date_position > 0:

        before_date = page_text[
            max(
                0,
                date_position - 150
            ):
            date_position
        ]

        before_date = clean_text(
            before_date
        )

        # Keep venue extraction conservative.
        # We can improve venue separately if needed.
        venue = before_date


    # -------------------------------------------------
    # Add verified row
    # -------------------------------------------------

    rows.append(
        {
            "date": match_date,
            "team1": team1,
            "team2": team2,
            "venue": venue,

            # Cricstats pages currently do not
            # reliably expose toss information.
            "toss_winner": pd.NA,
            "toss_decision": pd.NA,

            "match_type": "T20",
            "season": str(
                match_date.year
            ),
            "winner": winner,
            "source_url": match_url
        }
    )

    print(
        "INCLUDED:",
        match_date.date(),
        team1,
        "vs",
        team2,
        "Winner:",
        winner
    )

    time.sleep(
        0.3
    )


# -------------------------------------------------
# STEP 3
# Build DataFrame
# -------------------------------------------------

df = pd.DataFrame(
    rows
)

if df.empty:

    print(
        "\nERROR: No matches produced."
    )

    raise SystemExit


# -------------------------------------------------
# Deduplicate
# -------------------------------------------------

df = df.drop_duplicates(
    subset=[
        "date",
        "team1",
        "team2"
    ]
)

df = df.sort_values(
    "date"
).reset_index(
    drop=True
)


# -------------------------------------------------
# Final safety check
# -------------------------------------------------

bad_rows = df[
    ~(
        (df["team1"] == "Afghanistan")
        |
        (df["team2"] == "Afghanistan")
    )
]

if not bad_rows.empty:

    print(
        "\nERROR: Non-Afghanistan rows detected!"
    )

    print(
        bad_rows
    )

    raise SystemExit


# -------------------------------------------------
# Save
# -------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# -------------------------------------------------
# Summary
# -------------------------------------------------

print(
    "\n========================================"
)

print(
    "VERIFIED AFGHANISTAN T20I DATA"
)

print(
    "========================================"
)

print(
    "Total verified matches:",
    len(df)
)

print(
    "Earliest:",
    df["date"].min()
)

print(
    "Latest:",
    df["date"].max()
)

print(
    "\nMatches per year:"
)

print(
    df["date"]
    .dt.year
    .value_counts()
    .sort_index()
)

print(
    "\nMatches:"
)

print(
    df[
        [
            "date",
            "team1",
            "team2",
            "winner"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)