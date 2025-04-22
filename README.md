# league_winrate_model
Source code for Reddit post

Important note: There is a significant source of data leakage in that ranks of players is sampled at time of data collection - I addressed this by taking 20 fresh solo/duo games from a random player (https://op.gg/lol/summoners/na/Nightmyre-NA1?queue_type=SOLORANKED) and using the model on his, since opgg ranks are at time the game happened, not current ranks. The model got 15/20 right, which proves a correlation with one tailed p = 0.02. However, the likely actual correlation is less than the 77% advertised.
