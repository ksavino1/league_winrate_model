import os, re, numpy as np, pandas as pd
# Removed TensorFlow imports as NN model is not used
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# Removed ML preprocessing imports not needed for simple avg rank model
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report # Keep evaluation metrics

# ───────────────────────────── Rank → ELO helper ──────────────────────────────
# (Keep the get_elo_from_abbr function exactly as you provided it)
tier_bases = {
    "IRON": 0, "BRONZE": 400, "SILVER": 800, "GOLD": 1200,
    "PLATINUM": 1600, "EMERALD": 2000, "DIAMOND": 2400,
    "MASTER": 2800, "GRANDMASTER": 2800, "CHALLENGER": 2800
}
division_adds = {"4": 100, "3": 200, "2": 300, "1": 400}

def get_elo_from_abbr(rank):
    if rank is None or str(rank).lower() == "unranked" or pd.isna(rank):
        return None
    s = str(rank).strip().lower()

    long_tiers = {
        "iron": "IRON", "bronze": "BRONZE", "silver": "SILVER", "gold": "GOLD",
        "platinum": "PLATINUM", "emerald": "EMERALD", "diamond": "DIAMOND",
        "master": "MASTER", "grandmaster": "GRANDMASTER", "challenger": "CHALLENGER"
    }
    if s in long_tiers:
        # Handle single-word tiers (Master+)
        tier_full = long_tiers[s]
        if tier_full in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
             return tier_bases[tier_full]
        else: # Should not happen for valid single words
             return None

    m = re.fullmatch(r'([a-z]+)(\d)?', s)
    if not m:
        return None
    code, div = m.groups()
    tier_map = {
        "i": "IRON", "b": "BRONZE", "s": "SILVER", "g": "GOLD",
        "p": "PLATINUM", "e": "EMERALD", "d": "DIAMOND",
        "m": "MASTER", "gm": "GRANDMASTER", "c": "CHALLENGER"
    }
    tier = "GRANDMASTER" if code == "gm" else tier_map.get(code)
    if tier not in tier_bases:
        return None
    base = tier_bases[tier]
    # Apply division add ONLY if it's not Master+ AND div exists
    if tier not in ("MASTER", "GRANDMASTER", "CHALLENGER") and div:
         return base + division_adds.get(div, 0)
    else: # Return base for Master+ or if div is missing (shouldn't happen with regex)
         return base

# ─────────────────────────────── Hand‑typed data ───────────────────────────────
data_string = """
Game 1: blue wins
blue:
silver I
G4
p3
p3
p3

red:
p4
g3
unranked
p3
unranked

game 2: blue wins
p3
p4
p4
p3
p3

vs
p3
p2
p3
p4
p4

game 3: red wins
p4
g1
p4
p4
g1

vs

g2
p3
p3
p2
p4

game 4: blue side wins
p4
p4
p4
p4
p4

vs
g4
p4
p3
p4
p3

game 5: blue side wins
p4
p4
p4
p4
p4

vs

unranked
p4
p4
p4
p4


game 6: blue side wins
p4
s4
p4
s1
p3

vs

p4
p3
g2
s3
p4

game 7: red side wins
p4
p4
p4
p4
p3

vs

g3
p3
p3
p2
p3

game 8: blue side wins
p3
p4
p4
p4
p3

vs

p4
p4
p4
p4
p4

game 9: blue side wins
p4
p4
p4
p4
p4

vs
p4
p3
p4
g3
g2

game 10: red side wins
g2
p3
g2
g1
s2

vs
s1
s3
p4
g1
g1

game 11: red side wins
g2
p3
p4
p4
p4

vs

p3
p4
p4
p4
p4

game 12:  red side wins
p4
p4
p4
p4
p4

vs
p4
p2
p4
p4
g3

game 13: blue side wins
p3
p3
p3
p3
p4

vs

p4
p4
g4
p4
p3

game 14: red side wins
p3
p4
p3
p4
g2

vs
unranked
g1
p3
p3
p3

game 15: blue side wins
g2
p4
p2
p2
p3

vs
p3
p2
p3
g3
p4


game 16: red side wins
p3
p4
g1
p2
p2

vs
g1
p2
p1
p4
p2

game 17: blue side wins
p1
p3
p3
p2
p3

vs

g3
p2
p3
p3
p2

game 18: blue side wins
p4
p3
p2
p3
p3

vs

p3
p3
p3
g1
p3


game 19: red side wins
p3
p3
p2
p3
p3

vs
p2
p2
p3
p3
p2

game 20: blue side wins
p3
g4
p2
g3
p1

vs
s1
p2
p3
p4
p2

game 21: red side wins
p4
p3
p4
p4
p4

vs
p4
g1
p3
p3
p4

game 22: red side wins
unranked
g4
p4
p3
p4

vs
p3
p3
p4
p4
p3

game 23: red side wins
p3
unranked
p3
p3
p3

vs
p4
p3
p4
p3
p3

game 24: red side wins
p2
p3
p4
p3
p4

vs
p2
p2
p3
p2
p4

game 25: blue side wins
unranked
p3
g3
unranked
p3

vs
g2
g2
s2
p3
g2

game 26: red side wins
g2
p2
p4
p4
p4

vs
p3
g3
g4
p3
p4

game 27: blue side wins
g1
p4
p2
p3
p3

vs
p3
g2
p1
p4
p4

game 28: red side wins
p3
p3
g1
p3
p4

vs
g2
p3
p4
p3
p3

game 29: blue side wins
p4
p4
p4
p3
p4

vs
p4
p4
p4
p4
g1

game 30: blue side wins
p4
p4
p3
p4
p3

vs
p4
p4
p2
p3
g2
"""

# ──────────────────────────────── Parse the text ────────────────────────────────
# (Keep the parsing logic exactly as you provided it)
roles = ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]
games_data = []
pattern = re.compile(r'game\s+\d+\s*:', re.IGNORECASE)
match_positions = [m.start() for m in pattern.finditer(data_string)]
match_positions.append(len(data_string))
for i in range(len(match_positions) - 1):
    chunk = data_string[match_positions[i]:match_positions[i+1]].strip()
    try: # Add try block for robustness during parsing
        header, body = chunk.split(":", 1)
        winner_line = body.splitlines()[0].strip().lower()
        blue_win = 1 if ("blue" in winner_line and "win" in winner_line) else 0
        blue_raw, red_raw = [], []
        current = "blue"
        for line in body.splitlines()[1:]:
            l = line.strip()
            if not l: continue
            low = l.lower()
            if low in {"blue:", "blue"}: current = "blue"; continue
            if low in {"red:", "red", "vs"}: current = "red"; continue
            (blue_raw if current == "blue" else red_raw).append(l)
        if len(blue_raw) != 5 or len(red_raw) != 5:
            print(f"Warning: Malformed ranks in block {i+1}. Skipping. Blue: {len(blue_raw)}, Red: {len(red_raw)}")
            continue # Skip this game if counts are wrong
        row = {"blue_win": blue_win}
        for j, r in enumerate(roles):
            row[f"{r}_rank_blue"] = get_elo_from_abbr(blue_raw[j])
            row[f"{r}_rank_red"]  = get_elo_from_abbr(red_raw[j])
        games_data.append(row)
    except Exception as e:
        print(f"Error parsing block {i+1}: {e}")
        print(f"Problematic chunk: {chunk[:150]}...") # Print start of chunk for debug
        continue # Skip this game on error

if len(games_data) != 30:
     print(f"Warning: Parsed {len(games_data)} games, expected 30. Proceeding with available data.")
     if not games_data:
          print("No games parsed successfully. Exiting.")
          exit()


df_manual = pd.DataFrame(games_data)
print(f"Parsed {len(df_manual)} games into DataFrame.")

# ────────────────────────── Impute any missing ranks ────────────────────────────
# (Keep the imputation logic exactly as you provided it)
print("Imputing missing ranks...")
rank_cols = [f"{r}_rank_{side}" for side in ("blue", "red") for r in roles]
blue_rank_cols = [f"{r}_rank_blue" for r in roles]
red_rank_cols = [f"{r}_rank_red" for r in roles]

initial_nan_count = df_manual[rank_cols].isnull().sum().sum()
row_means = df_manual[rank_cols].mean(axis=1)
for c in rank_cols:
    df_manual[c] = df_manual[c].fillna(row_means)
if df_manual[rank_cols].isnull().values.any():
    # Calculate global mean only from non-NaN values in the original DataFrame
    global_mean_rank = df_manual[rank_cols].stack().mean()
    if pd.isna(global_mean_rank): global_mean_rank = 1200 # Fallback
    print(f"Applying global mean fallback: {global_mean_rank:.0f}")
    df_manual[rank_cols] = df_manual[rank_cols].fillna(global_mean_rank)

# Ensure conversion to int happens after ALL filling
df_manual[rank_cols] = df_manual[rank_cols].round().astype(int)
final_nan_count = df_manual[rank_cols].isnull().sum().sum()
print(f"Imputed {initial_nan_count - final_nan_count} missing rank values.")
print("Imputation complete.")

# ──────────────────── Apply Simple Avg Rank Model & Evaluate ────────────────────

print("\nCalculating average ranks per team...")
# Calculate average ranks per team using the imputed integer values
df_manual['avg_rank_blue'] = df_manual[blue_rank_cols].mean(axis=1)
df_manual['avg_rank_red'] = df_manual[red_rank_cols].mean(axis=1)

print("Predicting winner based on higher average rank...")
# Predict winner: Blue wins if their average rank is strictly higher
df_manual['prediction'] = (df_manual['avg_rank_blue'] > df_manual['avg_rank_red']).astype(int)

# --- Evaluate Accuracy ---
actual_wins = df_manual['blue_win']
predictions = df_manual['prediction']

accuracy = accuracy_score(actual_wins, predictions)
correct_count = (actual_wins == predictions).sum()
total_games = len(df_manual)

print(f"\n--- Accuracy of Simple Avg Rank Model on {total_games} Manual Games ---")
print(f"Prediction Rule: Blue wins if avg_rank_blue > avg_rank_red")
print(f"\nAccuracy: {accuracy:.4f} ({correct_count}/{total_games} correct)")

print("\nClassification Report:")
# Use zero_division=0 to avoid warnings if one class is never predicted
print(classification_report(actual_wins, predictions, target_names=["Blue loses", "Blue wins"], zero_division=0))

# Display average ranks for context
print("\nAverage Rank Stats (after imputation):")
print(f"  Avg Blue Team Rank: {df_manual['avg_rank_blue'].mean():.0f}")
print(f"  Avg Red Team Rank: {df_manual['avg_rank_red'].mean():.0f}")
print(f"  Avg Overall Player Rank: {df_manual[rank_cols].values.mean():.0f}")


print("\nScript finished.")