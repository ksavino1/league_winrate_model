import requests
import time
import random
import re
import json
from datetime import datetime
import math
import os
from dotenv import load_dotenv


# This is the main file used to populate our jsons, allowing it to run for extended periods of time (3-5 days) will speed up data
# acquisition due to caching.

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("API_KEY") # Key for pinging riot post match stats

# API Regions
match_region_url = "https://americas.api.riotgames.com"  # For match data
rank_region_url = "https://na1.api.riotgames.com"  # For rank data / summoner data

# Queue IDs to target
target_queue_ids = {
    400: "normal_draft_games",
    420: "ranked_solo_duo_games",
    440: "ranked_flex_games"
}

# Target number of players to fetch per TIER
TARGET_PLAYERS_PER_TIER = {
    "IRON": 0,
    "BRONZE": 0,
    "SILVER": 0,
    "GOLD": 1000,
    "PLATINUM": 1000,
    "EMERALD": 0,
    "DIAMOND": 0,
    "MASTER": 0,
    "GRANDMASTER": 0,
    "CHALLENGER": 0
}

# Maximum games to fetch per player across ALL queues
MAX_MATCHES_PER_PLAYER = 30  # Limit total games processed per player - promotes data variety
MAX_CONSECUTIVE_INVALID_GAMES = 0

headers = {"X-Riot-Token": API_KEY}

# Cache for Lolalytics matchup data to avoid re-scraping - allowing the script to run for longer will improve performance
seen_matchups = {}  # key = (champ1_url, champ2_url) --> winrate differential


def calculate_elo_score(tier, division, lp=0):
    """ Calculates a numerical ELO score based on tier, division, and LP. """
    tier_values = {
        "IRON": 0, "BRONZE": 400, "SILVER": 800, "GOLD": 1200,
        "PLATINUM": 1600, "EMERALD": 2000, "DIAMOND": 2400,
        "MASTER": 2800, "GRANDMASTER": 2800, "CHALLENGER": 2800
    }
    division_values = {"IV": 100, "III": 200, "II": 300, "I": 400}

    elo_score = tier_values.get(tier.upper(), 0)

    if tier.upper() in ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]:
        elo_score += division_values.get(division.upper(), 0)
    elif tier.upper() in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
        rounded_lp = round(lp / 100) * 100
        elo_score += rounded_lp
    return elo_score


def get_champion_matchup_data(champion1, champion2, role="top"):
    """
    Scrape the normalized win rate differential between two champions from Lolalytics.
    Uses regex and includes caching. NOTE: Web scraping is fragile.
    """
    global seen_matchups

    # Handle Riot's name for Wukong, he has a different name internally.
    if champion1 == "MonkeyKing": champion1 = "wukong"
    if champion2 == "MonkeyKing": champion2 = "wukong"

    # Format names for URL
    champion1_url = champion1.lower().replace(" ", "").replace("'", "").replace(".", "")
    champion2_url = champion2.lower().replace(" ", "").replace("'", "").replace(".", "")

    # Check cache first (both directions)
    if (champion1_url, champion2_url) in seen_matchups:
        return seen_matchups[(champion1_url, champion2_url)]
    if (champion2_url, champion1_url) in seen_matchups:
        # Return the inverse if the opposite matchup is cached
        return -1 * seen_matchups[(champion2_url, champion1_url)]

    # Map Riot roles to Lolalytics roles
    role_map = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "middle", "MID": "middle",
                "ADC": "bottom", "BOT": "bottom", "BOTTOM": "bottom",
                "SUPPORT": "support", "UTILITY": "support"}
    lolalytics_role = role_map.get(role.upper(), role.lower())

    # Construct Lolalytics URL
    url = f"https://lolalytics.com/lol/{champion1_url}/vs/{champion2_url}/build/?lane={lolalytics_role}&tier=all&vslane={lolalytics_role}"
    print(f"      Scraping Lolalytics: {url}")  # Add log for scraping attempts

    try:
        scrape_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=scrape_headers, timeout=15)  # Add timeout

        if response.status_code == 200:
            # Try primary regex pattern
            pattern = r"After normalising both champions win rates.+?([+-]?\d+\.\d+)%"
            match = re.search(pattern, response.text)
            if match:
                wr_differential = float(match.group(1))
                seen_matchups[(champion1_url, champion2_url)] = wr_differential
                print(f"      Scraped diff: {wr_differential}%")
                return wr_differential

            # Try fallback regex pattern
            pattern = r"([+-]?\d+\.\d+)% higher against"
            match = re.search(pattern, response.text)
            if match:
                wr_differential = float(match.group(1))
                seen_matchups[(champion1_url, champion2_url)] = wr_differential
                print(f"      Scraped diff (fallback): {wr_differential}%")
                return wr_differential

            print(
                f"      Could not extract WR diff for {champion1} vs {champion2} in {lolalytics_role} from page content.")
            return None
        else:
            print(f"      Failed Lolalytics scrape for {champion1} vs {champion2}: Status code {response.status_code}")
            return None

    except requests.exceptions.Timeout:
        print(f"      Timeout scraping Lolalytics for {champion1} vs {champion2}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"      Error scraping Lolalytics for {champion1} vs {champion2}: {e}")
        return None
    except Exception as e:
        print(f"      Unexpected error during Lolalytics scrape for {champion1} vs {champion2}: {e}")
        return None


def get_current_patch():
    """ Fetches the current patch version from Riot's Data Dragon. """
    versions_url = "https://ddragon.leagueoflegends.com/api/versions.json"
    try:
        versions_response = requests.get(versions_url, timeout=10)
        if versions_response.status_code == 200:
            versions = versions_response.json()
            if versions:
                current_patch_full = versions[0]
                major_minor_patch = ".".join(current_patch_full.split(".")[:2])
                # The hardcoded patch below is now commented out to use the fetched version
                # major_minor_patch = "15.6" # <-- REMOVED/COMMENTED OUT
                print(f"Current patch from API: {current_patch_full} (filtering for {major_minor_patch}.*)")
                return major_minor_patch
            else:
                print("Error: versions.json was empty.")
                return None
        else:
            print(f"Could not fetch current patch version, status code: {versions_response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching current patch: {e}")
        return None


def get_ranks_to_sample():
    """ Defines all Tiers/Divisions to sample players from. """
    return [
        # Iron - Diamond IV-I
        {"tier": "IRON", "division": "IV"}, {"tier": "IRON", "division": "III"}, {"tier": "IRON", "division": "II"},
        {"tier": "IRON", "division": "I"},
        {"tier": "BRONZE", "division": "IV"}, {"tier": "BRONZE", "division": "III"},
        {"tier": "BRONZE", "division": "II"}, {"tier": "BRONZE", "division": "I"},
        {"tier": "SILVER", "division": "IV"}, {"tier": "SILVER", "division": "III"},
        {"tier": "SILVER", "division": "II"}, {"tier": "SILVER", "division": "I"},
        {"tier": "GOLD", "division": "IV"}, {"tier": "GOLD", "division": "III"}, {"tier": "GOLD", "division": "II"},
        {"tier": "GOLD", "division": "I"},
        {"tier": "PLATINUM", "division": "IV"}, {"tier": "PLATINUM", "division": "III"},
        {"tier": "PLATINUM", "division": "II"}, {"tier": "PLATINUM", "division": "I"},
        {"tier": "EMERALD", "division": "IV"}, {"tier": "EMERALD", "division": "III"},
        {"tier": "EMERALD", "division": "II"}, {"tier": "EMERALD", "division": "I"},
        {"tier": "DIAMOND", "division": "IV"}, {"tier": "DIAMOND", "division": "III"},
        {"tier": "DIAMOND", "division": "II"}, {"tier": "DIAMOND", "division": "I"},
        # Master+ handled separately
        {"tier": "MASTER", "division": "I"},
        {"tier": "GRANDMASTER", "division": "I"},
        {"tier": "CHALLENGER", "division": "I"}
    ]


def get_players_by_rank():
    """ Fetches players based on TARGET_PLAYERS_PER_TIER using appropriate API endpoints. """
    player_list = []
    ranks_to_sample = get_ranks_to_sample()
    processed_players = 0

    print(f"Attempting to fetch players based on targets: {TARGET_PLAYERS_PER_TIER}")
    print("=" * 30)

    # --- Iron through Diamond (using /entries endpoint with pagination) ---
    tiers_with_divisions = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
    for tier in tiers_with_divisions:
        target_for_tier = TARGET_PLAYERS_PER_TIER.get(tier, 0)
        if target_for_tier == 0: continue

        target_per_division = math.ceil(target_for_tier / 4)
        print(f"\n--- Fetching for Tier: {tier} (Target: {target_for_tier}, ~{target_per_division} per division) ---",
              flush=True)

        for division in ["IV", "III", "II", "I"]:
            players_found_for_division = 0
            page = 1
            print(f"Fetching {tier} {division} (Target: {target_per_division})...")

            while players_found_for_division < target_per_division:
                print(f"  Requesting page {page} for {tier} {division}...")
                rank_url = f"{rank_region_url}/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}"
                params = {"page": page}

                try:
                    rank_response = requests.get(rank_url, headers=headers, params=params, timeout=15)

                    if rank_response.status_code == 429:  # Handle Rate Limiting
                        retry_after = int(rank_response.headers.get('Retry-After', 5))
                        print(f"  Rate limited fetching entries. Waiting {retry_after} seconds...")
                        time.sleep(retry_after + 1)
                        continue  # Retry the same page

                    if rank_response.status_code == 200:
                        rank_data = rank_response.json()
                        if not rank_data:
                            print(f"  No more players found for {tier} {division} on page {page}.")
                            break  # Stop fetching pages for this division

                        random.shuffle(rank_data)
                        print(f"  Found {len(rank_data)} players on page {page}.")

                        # Process players from this page
                        for entry in rank_data:
                            if players_found_for_division >= target_per_division: break

                            summoner_id = entry.get("summonerId")
                            summoner_name = entry.get("summonerName", "Unknown Name")  # Add default

                            if summoner_id:
                                # --- Get PUUID (Slow part) ---
                                try:
                                    print(
                                        f"    Fetching PUUID for {summoner_name} (ID: {summoner_id})...")  # Log PUUID fetch
                                    summoner_url = f"{rank_region_url}/lol/summoner/v4/summoners/{summoner_id}"
                                    summoner_response = requests.get(summoner_url, headers=headers, timeout=10)

                                    if summoner_response.status_code == 429:
                                        retry_after_puuid = int(summoner_response.headers.get('Retry-After', 3))
                                        print(
                                            f"    Rate limited fetching PUUID for {summoner_name}. Waiting {retry_after_puuid} sec...")
                                        time.sleep(retry_after_puuid + 1)
                                        summoner_response = requests.get(summoner_url, headers=headers,
                                                                         timeout=10)  # Retry

                                    if summoner_response.status_code == 200:
                                        summoner_data = summoner_response.json()
                                        puuid = summoner_data.get("puuid")
                                        if puuid:  # Ensure PUUID exists
                                            player_list.append({
                                                "puuid": puuid,
                                                "summoner_id": summoner_id,
                                                "summoner_name": summoner_name,
                                                "tier": tier,
                                                "division": division,
                                                "lp": entry.get("leaguePoints", 0)
                                            })
                                            players_found_for_division += 1
                                            processed_players += 1
                                            print(
                                                f"      Added player {processed_players}: {summoner_name} ({tier} {division}) PUUID: ...{puuid[-6:]}")
                                        else:
                                            print(
                                                f"    WARN: PUUID missing in response for {summoner_name} (ID: {summoner_id}). Skipping.")

                                    elif summoner_response.status_code == 404:
                                        print(
                                            f"    Summoner not found (ID: {summoner_id}, Name: {summoner_name}). Skipping.")
                                    else:
                                        print(
                                            f"    Failed to get PUUID for {summoner_name}: Status {summoner_response.status_code}")

                                    # --- ESSENTIAL Rate Limit Delay for PUUID lookup ---
                                    # This sleep respects the 100req/120sec limit for these individual lookups
                                    print("    Sleeping 1.21s after PUUID fetch...")
                                    time.sleep(1.21)

                                except requests.exceptions.RequestException as summoner_e:
                                    print(f"    Network error fetching PUUID for {summoner_name}: {summoner_e}")
                                    time.sleep(2)
                                except Exception as summoner_e:
                                    print(f"    Unexpected error fetching PUUID for {summoner_name}: {summoner_e}")
                                    time.sleep(1)

                            # Break inner loop if division target met
                            if players_found_for_division >= target_per_division: break

                        # Check if target met after processing page
                        if players_found_for_division >= target_per_division:
                            print(f"  Target of {target_per_division} met for {tier} {division}.")
                            break  # Stop fetching pages for this division

                        page += 1  # Go to next page

                    elif rank_response.status_code == 404:
                        print(f"  Rank/Division not found: {tier} {division}. Might be API issue.")
                        break
                    else:
                        print(f"  Failed to get players page {page} for {tier} {division}: {rank_response.status_code}")
                        print(f"  Response: {rank_response.text}")
                        time.sleep(5)
                        break  # Stop fetching for this division on error

                except requests.exceptions.RequestException as e_rank:
                    print(f"  Network error fetching {tier} {division} page {page}: {e_rank}. Retrying in 10s...")
                    time.sleep(10)
                except Exception as e_rank_unexp:
                    print(f"  Unexpected error fetching {tier} {division} page {page}: {e_rank_unexp}")
                    time.sleep(5)
                    break

            print(f"  Finished fetching for {tier} {division}. Found {players_found_for_division} players.")

    print("\nFinished fetching Iron through Diamond.")
    print("=" * 30)

    # --- Master, Grandmaster, Challenger ---
    for tier in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
        target_for_tier = TARGET_PLAYERS_PER_TIER.get(tier, 0)
        if target_for_tier == 0: continue

        print(f"\n--- Fetching for Tier: {tier} (Target: {target_for_tier}) ---")
        division = "I"
        league_endpoint_name = f"{tier.lower()}leagues"
        top_tier_url = f"{rank_region_url}/lol/league/v4/{league_endpoint_name}/by-queue/RANKED_SOLO_5x5"

        try:
            print(f"Fetching all {tier} entries...")
            tier_response = requests.get(top_tier_url, headers=headers, timeout=30)

            if tier_response.status_code == 429:
                retry_after = int(tier_response.headers.get('Retry-After', 10))
                print(f"  Rate limited fetching {tier} list. Waiting {retry_after} seconds...")
                time.sleep(retry_after + 1)
                tier_response = requests.get(top_tier_url, headers=headers, timeout=30)  # Retry

            if tier_response.status_code == 200:
                tier_data = tier_response.json()
                entries = tier_data.get("entries", [])
                if not entries:
                    print(f"  No players found in the {tier} league response.")
                    continue

                print(f"  Found {len(entries)} total players in {tier}.")
                num_to_sample = min(target_for_tier, len(entries))
                if num_to_sample < target_for_tier:
                    print(
                        f"  Warning: Only found {len(entries)} players, less than target {target_for_tier}. Taking all.")

                sampled_entries = random.sample(entries, num_to_sample)
                print(f"  Sampling {len(sampled_entries)} players...")
                players_added_this_tier = 0

                for entry in sampled_entries:
                    summoner_id = entry.get("summonerId")
                    summoner_name = entry.get("summonerName", "Unknown Name")  # Add default
                    lp = entry.get("leaguePoints", 0)

                    if summoner_id:
                        # --- Get PUUID (Slow) ---
                        try:
                            print(
                                f"    Fetching PUUID for {summoner_name} ({tier} - ID: {summoner_id})...")  # Log PUUID fetch
                            summoner_url = f"{rank_region_url}/lol/summoner/v4/summoners/{summoner_id}"
                            summoner_response = requests.get(summoner_url, headers=headers, timeout=10)

                            if summoner_response.status_code == 429:
                                retry_after_puuid = int(summoner_response.headers.get('Retry-After', 3))
                                print(
                                    f"    Rate limited fetching PUUID for {summoner_name} ({tier}). Waiting {retry_after_puuid} sec...")
                                time.sleep(retry_after_puuid + 1)
                                summoner_response = requests.get(summoner_url, headers=headers, timeout=10)  # Retry

                            if summoner_response.status_code == 200:
                                summoner_data = summoner_response.json()
                                puuid = summoner_data.get("puuid")
                                if puuid:  # Ensure PUUID exists
                                    player_list.append({
                                        "puuid": puuid,
                                        "summoner_id": summoner_id,
                                        "summoner_name": summoner_name,
                                        "tier": tier,
                                        "division": division,
                                        "lp": lp
                                    })
                                    players_added_this_tier += 1
                                    processed_players += 1
                                    print(
                                        f"      Added player {processed_players}: {summoner_name} ({tier} {division}, {lp} LP) PUUID: ...{puuid[-6:]}")
                                else:
                                    print(
                                        f"    WARN: PUUID missing in response for {summoner_name} ({tier} - ID: {summoner_id}). Skipping.")

                            elif summoner_response.status_code == 404:
                                print(f"    Summoner not found (ID: {summoner_id}, Name: {summoner_name}). Skipping.")
                            else:
                                print(
                                    f"    Failed to get PUUID for {summoner_name} ({tier}): Status {summoner_response.status_code}")

                            # --- ESSENTIAL Rate Limit Delay for PUUID lookup ---
                            print("    Sleeping 1.21s after PUUID fetch...")
                            time.sleep(1.21)

                        except requests.exceptions.RequestException as summoner_e:
                            print(f"    Network error fetching PUUID for {summoner_name} ({tier}): {summoner_e}")
                            time.sleep(2)
                        except Exception as summoner_e:
                            print(f"    Unexpected error fetching PUUID for {summoner_name} ({tier}): {summoner_e}")
                            time.sleep(1)

                print(f"  Finished sampling for {tier}. Added {players_added_this_tier} players.")

            elif tier_response.status_code == 404:
                print(f"  {tier} league endpoint not found ({top_tier_url}). Check API/URL.")
            else:
                print(
                    f"  Failed to get {tier} players: Status {tier_response.status_code}, Response: {tier_response.text}")
                time.sleep(5)

        except requests.exceptions.RequestException as e_league:
            print(f"  Network error fetching {tier} league: {e_league}")
            time.sleep(10)
        except Exception as e_league_unexp:
            print(f"  Unexpected error fetching {tier} league: {e_league_unexp}")
            time.sleep(5)

    print("=" * 30)
    print(f"Finished fetching all players. Total players collected: {len(player_list)}")
    print("=" * 30)
    return player_list


def get_player_matches(puuid, queue_id, start=0, count=20):  # Reduced count for smaller batches
    """ Get a batch of match IDs for a player for a specific queue. """
    if not puuid:
        print("  Cannot fetch matches, PUUID is missing.")
        return []

    match_ids_url = f"{match_region_url}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    # Fetch only for the specific queue ID
    params = {"start": start, "count": count, "queue": queue_id}
    print(f"  Fetching match IDs: start={start}, count={count}, queue={queue_id}")  # Log match ID fetch attempt

    try:
        response = requests.get(match_ids_url, headers=headers, params=params, timeout=15)

        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 5))
            print(f"  Rate limited getting match IDs. Waiting {retry_after} seconds...")
            time.sleep(retry_after + 1)
            response = requests.get(match_ids_url, headers=headers, params=params, timeout=15)  # Retry

        if response.status_code == 200:
            match_ids = response.json()
            print(f"  Found {len(match_ids)} match IDs in batch.")
            return match_ids
        elif response.status_code == 404:  # Often means no matches found for that queue/range
            print(f"  No matches found for PUUID ...{puuid[-6:]} with queue={queue_id}, start={start}.")
            return []
        elif response.status_code == 403:  # API Key issue
            print(f"  ERROR 403: Forbidden. Check API Key. PUUID: ...{puuid[-6:]}")
            return []  # Stop trying for this player probably
        else:
            print(f"  Failed to get matches for queue {queue_id}, PUUID ...{puuid[-6:]}: Status {response.status_code}")
            print(f"  Response: {response.text}")
            return []  # Treat as no matches found for this attempt

    except requests.exceptions.RequestException as e:
        print(f"  Network error getting matches for PUUID ...{puuid[-6:]}: {e}")
        return []
    except Exception as e_unexp:
        print(f"  Unexpected error getting matches for PUUID ...{puuid[-6:]}: {e_unexp}")
        return []


def process_match(match_id, reference_player_puuid, current_patch):
    """ Fetches and processes data for a specific match ID, including participant ranks and matchup differentials. """
    print(f"    Processing match: {match_id} for player {reference_player_puuid}")
    match_url = f"{match_region_url}/lol/match/v5/matches/{match_id}"

    try:
        match_response = requests.get(match_url, headers=headers, timeout=20)

        if match_response.status_code == 429:
            retry_after = int(match_response.headers.get('Retry-After', 5))
            print(f"    Rate limited processing match {match_id}. Waiting {retry_after} seconds...")
            time.sleep(retry_after + 1)
            match_response = requests.get(match_url, headers=headers, timeout=20)  # Retry

        if match_response.status_code != 200:
            print(f"    Error fetching match {match_id}: Status {match_response.status_code}")
            # Potentially log response text for debugging if needed:
            # try:
            #     print(f"    Response Text: {match_response.text[:200]}...") # Log first 200 chars
            # except Exception: pass # Ignore errors logging response text
            return None

        match_data = match_response.json()
        info = match_data.get("info", {})
        metadata = match_data.get("metadata", {})
        participants = info.get("participants", [])
        teams = info.get("teams", [])

        if not all([info, metadata, participants, teams]):
            print(f"    Match data for {match_id} missing info, metadata, participants, or teams. Skipping.")
            return None

        # --- Filter by Patch ---
        game_version = info.get("gameVersion", "")
        is_current_patch = game_version.startswith(
            current_patch + ".") if current_patch else False  # Check major.minor.
        if not is_current_patch:
            print(f"    Skipping game {match_id} from patch {game_version} (current filter: {current_patch}.*)")
            return None

        # --- Filter by Queue ID ---
        queue_id = info.get("queueId", None)
        if queue_id not in target_queue_ids:
            print(
                f"    Skipping game {match_id} with queue ID {queue_id} (not in target queues: {list(target_queue_ids.keys())})")
            return None

        # --- Find Reference Player & Team ---
        reference_participant = next((p for p in participants if p.get("puuid") == reference_player_puuid), None)
        if not reference_participant:
            print(
                f"    Reference player (PUUID: ...{reference_player_puuid[-6:]}) not found in match {match_id}. Skipping.")
            return None
        reference_team_id = reference_participant.get("teamId")

        # --- Determine Winning Team ---
        winning_team = next((team.get("teamId") for team in teams if team.get("win", False)), None)

        queue_type_key = target_queue_ids.get(queue_id, "unknown_queue")

        # --- Initialize Game Info Structure ---
        game_info = {
            "matchId": metadata.get("matchId"),
            "queueId": queue_id, "queueType": queue_type_key,
            "gameVersion": game_version, "winningTeam": winning_team,
            "referenceTeam": reference_team_id, "gameDuration": info.get("gameDuration"),
            "gameCreation": info.get("gameCreation"),
            "roles": {}, "champions": {}, "matchup_differentials": {}
        }

        position_map = {"TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MID",
                        "BOTTOM": "BOT", "UTILITY": "SUPPORT"}
        standard_roles = ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"]

        # Initialize structures
        for role in standard_roles:
            game_info["roles"][role] = {"blue": None, "red": None}
            game_info["champions"][role] = {"blue": None, "red": None}
            game_info["matchup_differentials"][role] = None

        # --- Process Each Participant (Get Rank and Champion) ---
        processed_puuids_for_rank = set()  # Avoid redundant rank lookups within this match

        for participant in participants:
            puuid = participant.get("puuid")
            if not puuid: continue

            team_id = participant.get("teamId")
            team_key = "blue" if team_id == 100 else "red" if team_id == 200 else None
            if not team_key: continue

            # Determine role
            riot_position = participant.get("teamPosition", "") or participant.get("individualPosition", "UNKNOWN")
            role = position_map.get(riot_position.upper())
            if not role or role not in standard_roles: continue  # Skip if role is invalid/unknown

            champion_name = participant.get("championName")
            summoner_id = participant.get("summonerId")

            # Store champion name immediately
            game_info["champions"][role][team_key] = champion_name

            # --- Get Rank Information (once per player per game) ---
            elo_score = None  # Default to None (unranked or error)

            if summoner_id and puuid not in processed_puuids_for_rank:
                processed_puuids_for_rank.add(puuid)
                print(f"      Fetching rank for player PUUID ...{puuid[-6:]} (SummonerID: {summoner_id})")
                rank_url = f"{rank_region_url}/lol/league/v4/entries/by-summoner/{summoner_id}"
                try:
                    rank_response = requests.get(rank_url, headers=headers, timeout=10)

                    if rank_response.status_code == 429:
                        retry_after_rank = int(
                            rank_response.headers.get('Retry-After', 3))  # Use slightly longer default
                        print(
                            f"      Rate limited fetching rank for ...{puuid[-6:]}. Waiting {retry_after_rank} sec...")
                        time.sleep(retry_after_rank + 1)
                        rank_response = requests.get(rank_url, headers=headers, timeout=10)  # Retry

                    if rank_response.status_code == 200:
                        rank_data = rank_response.json()
                        # Find RANKED_SOLO_5x5 entry
                        solo_duo_entry = next((q for q in rank_data if q.get("queueType") == "RANKED_SOLO_5x5"), None)
                        if solo_duo_entry:
                            tier = solo_duo_entry.get("tier", "UNRANKED")
                            division = solo_duo_entry.get("rank", "")  # Division is 'rank' field
                            lp = solo_duo_entry.get("leaguePoints", 0)
                            if tier != "UNRANKED":
                                elo_score = calculate_elo_score(tier, division, lp)
                                print(f"        Rank found: {tier} {division} {lp}LP -> ELO: {elo_score}")
                            else:
                                print(f"        Rank found: UNRANKED")
                        else:
                            print(f"        No RANKED_SOLO_5x5 entry found for ...{puuid[-6:]}")

                    elif rank_response.status_code == 404:  # Player is unranked or summoner ID invalid?
                        print(f"        No rank entries found (404) for summoner {summoner_id}")
                    elif rank_response.status_code == 403:
                        print(f"        ERROR 403: Forbidden fetching rank for summoner {summoner_id}. Check API Key.")
                    else:  # Other errors
                        print(
                            f"      Warn: Non-200/404 status ({rank_response.status_code}) getting rank for summoner {summoner_id}")

                    # *** CRITICAL RATE LIMIT FIX ***
                    # Apply sleep *after every rank lookup attempt* (success or fail)
                    # 1.21 seconds aims to respect 100 req / 120 seconds
                    print(f"      Sleeping 1.21s after rank fetch for ...{puuid[-6:]}")
                    time.sleep(1.21)

                except requests.exceptions.RequestException as e_rank:
                    print(f"      Network error getting rank for summoner {summoner_id}: {e_rank}")
                    time.sleep(1)  # Wait a bit after network error before next player
                except Exception as e_rank_unexp:
                    print(f"      Unexpected error getting rank for summoner {summoner_id}: {e_rank_unexp}")
                    time.sleep(1)

            # Store elo score (is None if unranked/error/not looked up yet)
            game_info["roles"][role][team_key] = elo_score
            # End of participant loop

        # --- Calculate Win Rate Differentials (after processing all participants) ---
        print(f"    Calculating matchup differentials for match {match_id}...")
        for role in standard_roles:
            blue_champion = game_info["champions"][role]["blue"]
            red_champion = game_info["champions"][role]["red"]

            if blue_champion and red_champion:
                print(f"      Getting diff for {role}: {blue_champion} (blue) vs {red_champion} (red)")
                differential = get_champion_matchup_data(blue_champion, red_champion, role)
                game_info["matchup_differentials"][role] = differential  # Store None if scraping failed

                # Add delay between Lolalytics scraping attempts (politeness)
                scrape_delay = random.uniform(1.5, 3.0)
                print(f"      Sleeping {scrape_delay:.2f}s after Lolalytics scrape...")
                time.sleep(scrape_delay)
            else:
                print(f"      Skipping diff for {role} (missing blue or red champ).")
                game_info["matchup_differentials"][role] = None  # Ensure it's None if matchup incomplete

        print(f"    Finished processing match {match_id}")
        return game_info

    # --- Outer Error Handling for process_match ---
    except requests.exceptions.RequestException as e_match:
        print(f"    Network error processing match {match_id}: {e_match}")
        return None
    except Exception as e:
        print(f"    Unexpected error processing match {match_id}: {e}")
        # import traceback # Uncomment for detailed debug logs if needed
        # print(traceback.format_exc()) # Uncomment for detailed debug logs if needed
        return None


# --- MAIN EXECUTION ---
def main():
    start_time = time.time()
    print("--- Starting Data Collection ---")

    # Initialize data structure
    all_games_data = {
        "ranked_solo_duo_games": {},
        "ranked_flex_games": {},
        "normal_draft_games": {}
    }

    # Get current patch
    current_patch = get_current_patch()
    if not current_patch:
        print("CRITICAL: Failed to get current patch. Cannot filter games by patch. Exiting.")
        return

    # Check API Key placeholder
    if "RGAPI-YOUR-API-KEY-HERE" in API_KEY or not API_KEY:
        print("CRITICAL: API Key seems invalid or is the placeholder. Please update it in the script. Exiting.")
        return

    # Get players
    print("Fetching players (this will take a while due to PUUID lookups and rate limits)...")
    player_list = get_players_by_rank()

    if not player_list:
        print("No players were found or fetched. Exiting.")
        return

    print(f"\n--- Starting Match Fetching for {len(player_list)} players ---")
    processed_players_count = 0
    total_games_added = 0
    processed_match_ids = set()  # Keep track of matches already processed globally

    # Iterate through players
    for player in player_list:
        processed_players_count += 1
        puuid = player.get("puuid")
        summoner_name = player.get("summoner_name", "N/A")
        player_tier = player.get("tier", "unknown").lower()
        player_division = player.get("division", "unknown").lower()
        rank_key = f"{player_tier} {player_division}"  # For data structure

        if not puuid:
            print(f"Skipping player {summoner_name} (Index: {processed_players_count - 1}) due to missing PUUID.")
            continue

        print(
            f"\n({processed_players_count}/{len(player_list)}) Processing Player: {summoner_name} ({rank_key}) PUUID: ...{puuid[-6:]}")
        player_games_added_this_run = 0  # Track games added for *this player* in *this run*

        # Get matches for each target queue type for this player
        for queue_id, queue_type_key in target_queue_ids.items():

            # Check if we've hit the max matches for this player across all queues during this run
            if player_games_added_this_run >= MAX_MATCHES_PER_PLAYER:
                print(
                    f"  Reached max games ({MAX_MATCHES_PER_PLAYER}) for player {summoner_name}, skipping remaining queues.")
                break  # Stop processing queues for this player

            print(f"  Checking queue: {queue_type_key} (ID: {queue_id})")

            # Fetch match history in batches for this queue
            start_index = 0
            batch_size = 20  # Fetch 20 match IDs at a time
            matches_in_queue_found = 0  # Count matches found for *this specific queue* for the player
            MAX_CONSECUTIVE_INVALID_GAMES = 0

            while True:  # Loop for fetching batches of match IDs
                # Check again before fetching next batch if max player games reached
                if player_games_added_this_run >= MAX_MATCHES_PER_PLAYER or MAX_CONSECUTIVE_INVALID_GAMES > 2:
                    print("Ending the queue for player.")
                    MAX_CONSECUTIVE_INVALID_GAMES = 0
                    break  # Stop processing this batch

                match_ids_batch = get_player_matches(puuid, queue_id, start=start_index, count=batch_size)

                # Add delay AFTER fetching match IDs list to space out API calls
                # Helps prevent bursting if match list + first match processing happens too fast
                print(f"  Sleeping 1.21s after fetching match ID batch...")
                time.sleep(1.21)

                if not match_ids_batch:
                    print(
                        f"  No more matches found for {summoner_name} in {queue_type_key} starting at index {start_index}.")
                    break  # Stop fetching batches for this queue

                # Process the batch of match IDs
                for match_id in match_ids_batch:
                    # Check if max player games reached before processing this match
                    if player_games_added_this_run >= MAX_MATCHES_PER_PLAYER or MAX_CONSECUTIVE_INVALID_GAMES > 2:
                        print("Ending this player.")
                        break  # Stop processing this batch

                    # Check if match was already processed (from another player's history perhaps)
                    if match_id in processed_match_ids:
                        print(f"    Skipping match {match_id} (already processed).")
                        continue

                    # Process the individual match (this function contains heavy rate limiting delays now)
                    game_info = process_match(match_id, puuid, current_patch)

                    if game_info:
                        # Add game to structure if valid
                        # Ensure tier exists
                        if player_tier not in all_games_data[queue_type_key]:
                            all_games_data[queue_type_key][player_tier] = {}
                        # Ensure rank_key exists
                        if rank_key not in all_games_data[queue_type_key][player_tier]:
                            all_games_data[queue_type_key][player_tier][rank_key] = {}

                        # Add the match data
                        all_games_data[queue_type_key][player_tier][rank_key][match_id] = game_info
                        processed_match_ids.add(match_id)  # Mark as processed globally
                        total_games_added += 1
                        player_games_added_this_run += 1
                        matches_in_queue_found += 1
                        print(
                            f"    ++++ Added game {match_id} to {queue_type_key} ({rank_key}). Player total this run: {player_games_added_this_run}. Global total: {total_games_added} ++++")
                    else:
                        # process_match returned None (e.g., wrong patch, API error, incomplete data)
                        # The function process_match already prints reasons for skipping
                        MAX_CONSECUTIVE_INVALID_GAMES += 1
                        if MAX_CONSECUTIVE_INVALID_GAMES >= 2:
                            print("Max conseuctive invalid games of ", MAX_CONSECUTIVE_INVALID_GAMES, " reached.")
                            break

                    # No sleep needed here - process_match handles sleeps after its internal calls
                    # and the main loop has a sleep after fetching the match ID list batch.

                # Break inner loop if max player games reached
                if player_games_added_this_run >= MAX_MATCHES_PER_PLAYER:
                    break

                # Update start index for the next batch of match IDs
                start_index += len(match_ids_batch)

                # If we received fewer matches than requested, it means we reached the end of history for this queue
                if len(match_ids_batch) < batch_size:
                    print(f"  Reached end of available match history for {queue_type_key}.")
                    break  # Stop fetching batches for this queue

            print(
                f"  Finished checking {queue_type_key} for {summoner_name}. Found {matches_in_queue_found} new valid games in this queue.")
            # Small delay between checking different queues for the same player
            # time.sleep(2) # Optional: Small delay between queues

        print(
            f"Finished all queues for {summoner_name}. Added {player_games_added_this_run} games total for this player.")
        # Optional: Longer delay between players if needed
        # time.sleep(5)

    print("\n--- Data Collection Complete ---")

    # Save data to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"league_match_data_{timestamp}.json"
    try:
        print(f"Saving data to {filename}...")
        with open(filename, 'w') as f:
            json.dump(all_games_data, f, indent=2)
        print(f"Data saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving data to JSON file: {e}")

    # Print summary
    print("\n--- Summary ---")
    print(f"Total players processed: {processed_players_count}/{len(player_list)}")
    print(
        f"Total unique games collected and saved: {total_games_added} (across {len(processed_match_ids)} unique match IDs)")

    games_by_queue = {qt: 0 for qt in all_games_data}
    for queue_type, tiers_data in all_games_data.items():
        queue_total = 0
        for tier, ranks_data in tiers_data.items():
            for rank_key, matches in ranks_data.items():
                queue_total += len(matches)
        games_by_queue[queue_type] = queue_total

    for queue_type, count in games_by_queue.items():
        print(
            f"  {queue_type}: {count} games stored")  # Note: this counts stored game entries, should match total_games_added

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes / {elapsed_time / 3600:.2f} hours)")


if __name__ == "__main__":
    MAX_CONSECUTIVE_INVALID_GAMES = 0
    main()