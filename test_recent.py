"""
test_recent.py  ▸  evaluate FF‑NN on 20 handwritten games
"""

import os, re, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

# ───────────────────────────── Rank → ELO helper ──────────────────────────────
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
        return tier_bases[long_tiers[s]]

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
    return base if tier in ("MASTER", "GRANDMASTER", "CHALLENGER") else base + division_adds.get(div, 0)

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
"""

# ──────────────────────────────── Parse the text ────────────────────────────────
roles = ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]
games_data = []

# find every "game X:" position so we can slice reliably even with odd spacing
pattern = re.compile(r'game\s+\d+\s*:', re.IGNORECASE)
match_positions = [m.start() for m in pattern.finditer(data_string)]
match_positions.append(len(data_string))               # sentinel end‑idx

for i in range(len(match_positions) - 1):
    chunk = data_string[match_positions[i]:match_positions[i+1]].strip()
    header, body = chunk.split(":", 1)
    winner_line = body.splitlines()[0].strip().lower()
    blue_win = 1 if ("blue" in winner_line and "win" in winner_line) else 0

    # collect raw rank strings
    blue_raw, red_raw = [], []
    current = "blue"
    for line in body.splitlines()[1:]:
        l = line.strip()
        if not l:
            continue
        low = l.lower()
        if low in {"blue:", "blue"}:
            current = "blue"; continue
        if low in {"red:", "red", "vs"}:
            current = "red";  continue
        (blue_raw if current == "blue" else red_raw).append(l)

    if len(blue_raw) != 5 or len(red_raw) != 5:
        raise RuntimeError(f"Malformed ranks in block:\n{chunk[:120]}...")

    row = {"blue_win": blue_win}
    for j, r in enumerate(roles):
        row[f"{r}_rank_blue"] = get_elo_from_abbr(blue_raw[j])
        row[f"{r}_rank_red"]  = get_elo_from_abbr(red_raw[j])
    games_data.append(row)

if len(games_data) != 20:
    raise ValueError(f"Expected 20 games; parsed {len(games_data)}")

df_manual = pd.DataFrame(games_data)
print("Parsed OK — 20 games.")

# ────────────────────────── Impute any missing ranks ────────────────────────────
rank_cols = [f"{r}_rank_{side}" for side in ("blue", "red") for r in roles]
row_means = df_manual[rank_cols].mean(axis=1)
for c in rank_cols:
    df_manual[c] = df_manual[c].fillna(row_means)
if df_manual[rank_cols].isnull().values.any():
    df_manual[rank_cols] = df_manual[rank_cols].fillna(df_manual[rank_cols].stack().mean())
df_manual[rank_cols] = df_manual[rank_cols].round().astype(int)

# ─────────────────────────────── Pre‑processing  ────────────────────────────────
df_manual["queueType"] = "ranked_solo_duo_games"
feat_cols = rank_cols + ["queueType"]
preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["queueType"]),
     ("num", StandardScaler(), rank_cols)]
)

X_manual = preprocessor.fit_transform(df_manual[feat_cols])
y_manual = df_manual["blue_win"].values

# ───────────────────────────────── Load model ────────────────────────────────────
model_file = "ff_nn_acc=0.7665369510650635_queue=ranked_solo_duo_games_champs=False_ranks=True_diffs=False.keras"
if not os.path.isfile(model_file):
    raise FileNotFoundError(model_file)
model = load_model(model_file)

# ───────────────────────────────── Evaluate ─────────────────────────────────────
probs = model.predict(X_manual)
preds = (probs > 0.5).astype(int).flatten()
acc = accuracy_score(y_manual, preds)
correct = (preds == y_manual).sum()

print(f"\nAccuracy on handwritten sample: {acc:.3f} ({correct}/20 correct)\n")
print(classification_report(y_manual, preds, target_names=["Blue loses", "Blue wins"]))
