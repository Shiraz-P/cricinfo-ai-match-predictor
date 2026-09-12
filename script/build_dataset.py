import json
import os
import pandas as pd

data_folder = r"K:\Python\Cricinfo_AI_Project\data\t20s_male_json"

rows = []

for filename in os.listdir(data_folder):

    if filename.endswith(".json"):

        file_path = os.path.join(data_folder, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        info = data["info"]
        teams = info["teams"]

        row = {
            "date": info["dates"][0],
            "team1": teams[0],
            "team2": teams[1],
            "venue": info.get("venue", "Unknown"),
            "toss_winner": info["toss"]["winner"],
            "toss_decision": info["toss"]["decision"],
            "match_type": info["match_type"],
            "season": info["season"],
            "winner": info.get("outcome", {}).get("winner", "No Result")
        }

        rows.append(row)

df = pd.DataFrame(rows)

output_file = r"K:\Python\Cricinfo_AI_Project\data\matches.csv"

df.to_csv(output_file, index=False)

print("Total matches:", len(df))
print("CSV created:", output_file)

print("\nFirst 5 rows:")
print(df.head())