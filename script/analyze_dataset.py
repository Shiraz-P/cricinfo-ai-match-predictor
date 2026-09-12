import pandas as pd

file_path = r"K:\Python\Cricinfo_AI_Project\data\matches.csv"

df = pd.read_csv(file_path)

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique teams:")
print(df["team1"].nunique())

print("\nUnique venues:")
print(df["venue"].nunique())

print("\nWinner counts:")
print(df["winner"].value_counts().head(10))

print("\nNo Result matches:")
print((df["winner"] == "No Result").sum())

print("\nDate range:")
print("Oldest:", df["date"].min())
print("Newest:", df["date"].max())


print("\n--- DATA CLEANING ---")

df_clean = df[df["winner"] != "No Result"].copy()

print("Original matches:", len(df))
print("No Result removed:", len(df) - len(df_clean))
print("Usable completed matches:", len(df_clean))

df_clean["team1_won"] = (
    df_clean["winner"] == df_clean["team1"]
).astype(int)

print("\nTarget distribution:")
print(df_clean["team1_won"].value_counts())

clean_file = r"K:\Python\Cricinfo_AI_Project\data\matches_clean.csv"

df_clean.to_csv(clean_file, index=False)

print("\nClean dataset created:")
print(clean_file)