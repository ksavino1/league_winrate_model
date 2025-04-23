# league_winrate_model
Source code for Reddit post

Important note: There is a significant source of data leakage in that ranks of players is sampled at time of data collection - I addressed this by taking 30 fresh solo/duo games from two random players (https://op.gg/lol/summoners/na/Nightmyre-NA1?queue_type=SOLORANKED, https://op.gg/lol/summoners/na/Old%20Man%20Lustaf-NA1?queue_type=SOLORANKED) and using the model on theirs, since opgg ranks are at time the game happened, not current ranks. The model got 23/30 right, which proves a correlation with one tailed p = 0.002611. However, the likely actual correlation could definitely be less than 77% as mentioned.
