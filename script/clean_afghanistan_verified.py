import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

INPUT_FILE = PROJECT_ROOT / "data" / "matches_afghanistan_verified.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "matches_afghanistan_final.csv"

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print("Before cleaning:", len(df))

# Afghanistan withdrew from the Nov 2025 Pakistan tri-series.
# These Cricstats records are invalid source records.
invalid_dates = pd.to_datetime([
    "2025-11-17",
    "2025-11-23",
    "2025-11-25"
])

invalid_rows = df[df["date"].isin(invalid_dates)]

print("\nInvalid rows being removed:")
print(
    invalid_rows[
        ["date", "team1", "team2", "winner"]
    ].to_string(index=False)
)

df = df[
    ~df["date"].isin(invalid_dates)
].copy()

df = df.sort_values("date").reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nAfter cleaning:", len(df))
print("Earliest:", df["date"].min())
print("Latest:", df["date"].max())

print("\nMatches per year:")
print(
    df["date"]
    .dt.year
    .value_counts()
    .sort_index()
)

print("\nSaved:")
print(OUTPUT_FILE)