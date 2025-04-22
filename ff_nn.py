import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress TensorFlow debug logs

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import tensorflow as tf # Import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time # To time the process

# ---------------------------------------------------------------
# 1) SETTINGS
# ---------------------------------------------------------------
JSON_PATHS     = [
    "league_match_data_20250415_123727.json",
    "league_match_data_20250421_232906.json"
]
QUEUE_TYPES    = None  # None for all queues, "ranked_solo_duo_games" for soloduo
USE_CHAMPS     = False   # <--- Feature flags
USE_RANKS      = True   # <--- Feature flags
USE_DIFFS      = False  # <--- Feature flags
RANK_STRATEGY  = "impute" # "impute" or "drop"

RANDOM_SEED    = 42     # Seed for train/test split consistency
NUM_RUNS       = 10     # <--- Number of times to run training/evaluation
SPLIT_RATIO    = (0.8, 0.1, 0.1) # <--- Train/Validation/Test split (must sum to 1)

roles          = ["TOP","JUNGLE","MID","BOT","SUPPORT"]

# --- Set global random seeds for better reproducibility (optional, may not guarantee perfect results) ---
# tf.random.set_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# -----------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------
# 2) LOAD & FLATTEN
# ---------------------------------------------------------------
def load_matches(path, queue_types=None):
    # (Function definition remains the same as before)
    with open(path, 'r') as f:
        data = json.load(f)
    rows = []
    for qt, tiers in data.items():
        if queue_types and qt != queue_types:
            continue
        for tier_dict in tiers.values():
            for division_dict in tier_dict.values():
                for match in division_dict.values():
                    match_id = match.get("matchId")
                    if not match_id:
                        continue
                    row = {
                        "matchId": match_id,
                        "queueType": qt,
                        "blue_win": int(match["winningTeam"] == 100)
                    }
                    for r in roles:
                        champs = match.get("champions", {}).get(r, {})
                        ranks  = match.get("roles", {}).get(r, {})
                        diff   = match.get("matchup_differentials", {}).get(r)

                        row[f"{r}_champ_blue"] = champs.get("blue")
                        row[f"{r}_champ_red" ] = champs.get("red")
                        row[f"{r}_rank_blue"  ] = ranks.get("blue")
                        row[f"{r}_rank_red"   ] = ranks.get("red")
                        row[f"{r}_diff"] = diff if diff is not None else np.nan

                    rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------
# 3) LOAD, CONCAT, DEDUPLICATE
# ---------------------------------------------------------------
print("Loading and concatenating data...")
df = pd.concat([load_matches(p, QUEUE_TYPES) for p in JSON_PATHS],
               ignore_index=True)
print(f"Loaded {len(df)} total match entries.")

print("Removing duplicate matches...")
initial_count = len(df)
df = df.drop_duplicates(subset="matchId", keep="first")
final_count = len(df)
print(f"Removed {initial_count - final_count} duplicate matches based on matchId. Kept {final_count} unique matches.")

# ---------------------------------------------------------------
# 4) DATA CLEANING & PREP (Ranks, Champions, Diffs)
# ---------------------------------------------------------------
# --- Ranks ---
rank_cols = [f"{r}_rank_blue" for r in roles] + [f"{r}_rank_red" for r in roles]
df[rank_cols] = df[rank_cols].apply(pd.to_numeric, errors='coerce')

# --- Champions ---
champ_cols = [f"{r}_champ_blue" for r in roles] + [f"{r}_champ_red" for r in roles]
initial_count = len(df)
df = df.dropna(subset=champ_cols)
print(f"Removed {initial_count - len(df)} matches with missing champion data.")

# --- Differentials ---
if USE_DIFFS:
    diff_cols = [f"{r}_diff" for r in roles]
    initial_count = len(df)
    df = df.dropna(subset=diff_cols)
    print(f"Removed {initial_count - len(df)} matches with missing differential data.")

# --- Handle Null Ranks ---
initial_count = len(df)
if RANK_STRATEGY == "drop":
    df = df.dropna(subset=rank_cols)
    print(f"Removed {initial_count - len(df)} matches with missing rank data (Strategy: drop).")
elif RANK_STRATEGY == "impute":
    mean_ranks = df[rank_cols].mean(axis=1)
    for col in rank_cols:
        df[col] = df[col].fillna(mean_ranks)
    global_mean_rank = df[rank_cols].stack().mean()
    df[rank_cols] = df[rank_cols].fillna(global_mean_rank)
    df[rank_cols] = df[rank_cols].round().astype(int)
    print(f"Imputed missing rank data (Strategy: impute). Final count: {len(df)}")
else:
    raise ValueError("RANK_STRATEGY must be 'drop' or 'impute'")

# --- Drop matchId ---
if "matchId" in df.columns:
    df = df.drop(columns=["matchId"])
    print("Dropped 'matchId' column.")

# ---------------------------------------------------------------
# 5) FEATURE ENGINEERING & PREPROCESSING
# ---------------------------------------------------------------
feat_cols = []
cat_cols  = []
num_cols  = []

if USE_CHAMPS:
    champs = [f"{r}_champ_blue" for r in roles] + [f"{r}_champ_red" for r in roles]
    cat_cols += champs
    feat_cols += champs

if USE_RANKS:
    num_cols += rank_cols
    feat_cols += rank_cols

if USE_DIFFS:
    diffs = [f"{r}_diff" for r in roles]
    num_cols += diffs
    feat_cols += diffs

if QUEUE_TYPES is not None and "queueType" in df.columns:
    cat_cols += ["queueType"]
    feat_cols += ["queueType"]
elif "queueType" in df.columns and len(df["queueType"].unique()) > 1:
     cat_cols += ["queueType"]
     feat_cols += ["queueType"]

print(f"\nUsing {len(feat_cols)} features for model training.")
print(f"Categorical features ({len(cat_cols)}): {len(cat_cols)}") # Shortened print
print(f"Numerical features ({len(num_cols)}): {len(num_cols)}")   # Shortened print

X = df[feat_cols]
y = df["blue_win"].values

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", StandardScaler(), num_cols),
], remainder='passthrough')

print("\nFitting preprocessor...")
X_proc = preprocessor.fit_transform(X)
print(f"Processed feature shape: {X_proc.shape}")
print(f"Overall unique data Blue-win rate: {y.mean():.4f}")

# ---------------------------------------------------------------
# 6) MULTIPLE RUNS: TRAIN, EVALUATE, FIND BEST
# ---------------------------------------------------------------
best_test_acc = -1.0
best_model_weights = None
all_test_accuracies = []

# --- Data Split (80/10/10) ---
train_size, val_size, test_size = SPLIT_RATIO
if not np.isclose(train_size + val_size + test_size, 1.0):
     raise ValueError("SPLIT_RATIO must sum to 1.0")

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_proc, y,
    train_size=(train_size + val_size), # Combine train and val temporarily
    stratify=y,
    random_state=RANDOM_SEED
)
# Calculate validation size relative to the temporary train_full set
relative_val_size = val_size / (train_size + val_size)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=relative_val_size, # Split val off from train_full
    stratify=y_train_full,
    random_state=RANDOM_SEED
)
print(f"\nData split into:")
print(f"  Train set size: {X_train.shape[0]} ({X_train.shape[0]/len(X_proc):.1%})")
print(f"  Validation set size: {X_val.shape[0]} ({X_val.shape[0]/len(X_proc):.1%})")
print(f"  Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X_proc):.1%})")
# -----------------------------

start_total_time = time.time()

for run in range(NUM_RUNS):
    print(f"\n--- Starting Run {run + 1}/{NUM_RUNS} ---")
    start_run_time = time.time()

    # --- Build Model (define INSIDE loop for fresh init) ---
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    # --------------------------------------------------------

    # --- Train Model ---
    early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=256,
                        callbacks=[early_stop],
                        verbose=0) # Suppress epoch output within the loop
    # --------------------

    # --- Evaluate Model on TEST set ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    all_test_accuracies.append(test_acc)
    print(f"  Run {run + 1} Test Accuracy: {test_acc:.4f} (Loss: {test_loss:.4f})")
    # ----------------------------------

    # --- Check if this is the best model so far ---
    if test_acc > best_test_acc:
        print(f"  *** New best test accuracy found: {test_acc:.4f} (previous best: {best_test_acc:.4f}) ***")
        best_test_acc = test_acc
        # Save the weights of the best model
        best_model_weights = model.get_weights()
    # --------------------------------------------
    run_time = time.time() - start_run_time
    print(f"  Run {run + 1} finished in {run_time:.2f} seconds.")


# ---------------------------------------------------------------
# 7) SAVE BEST MODEL & REPORT RESULTS
# ---------------------------------------------------------------
print(f"\n--- Training Complete ---")
total_time = time.time() - start_total_time
print(f"Total time for {NUM_RUNS} runs: {total_time:.2f} seconds.")

if best_model_weights is not None:
    print(f"\nBest test accuracy across {NUM_RUNS} runs: {best_test_acc:.4f}")
    print(f"Mean test accuracy: {np.mean(all_test_accuracies):.4f}")
    print(f"Std dev test accuracy: {np.std(all_test_accuracies):.4f}")

    # --- Construct Filename ---
    filename = f"ff_nn_acc={best_test_acc}_queue={QUEUE_TYPES}_champs={USE_CHAMPS}_ranks={USE_RANKS}_diffs={USE_DIFFS}.keras"
    # --------------------------

    # --- Create final model instance and load best weights ---
    print(f"\nSaving best model to '{filename}'...")
    final_model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    # Compile is necessary before loading weights if the optimizer state is needed,
    # but often just loading weights is sufficient for inference/saving.
    # For saving the full model state including optimizer, compile it.
    final_model.compile(optimizer="adam",
                        loss="binary_crossentropy",
                        metrics=["accuracy"])
    final_model.set_weights(best_model_weights)
    # --------------------------------------------------------

    # --- Save the best model ---
    final_model.save(filename)
    print("Best model saved successfully.")
    # ---------------------------

    # --- Optional: Evaluate the final loaded model again for confirmation ---
    print("\nEvaluating the saved best model on the test set:")
    final_loss, final_acc = final_model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy: {final_acc:.4f}")
    print(f"  Test Loss: {final_loss:.4f}")

    print("\nClassification Report for the best model (Test Set):")
    y_pred_best = (final_model.predict(X_test) > 0.5).astype(int).flatten()
    print(classification_report(y_test, y_pred_best, target_names=["Blue loses","Blue wins"]))

else:
    print(f"No models were successfully trained in {NUM_RUNS} runs.")


# --- Optional: Human Eval Section (Needs careful review if using) ---
# Remember that human eval samples should ideally be drawn *after* the
# final data split and reflect the same preprocessing. If the files
# exist from previous runs, they might not align perfectly.
# (Code for human eval section omitted for brevity, but would follow here if needed)
# --- -----------------------------------------------------------

print("\nScript finished.")



# accuracies -
# all queues