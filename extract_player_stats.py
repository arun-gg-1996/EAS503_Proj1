import json
import os
import time
from pprint import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import custom modules (these would be local imports in your actual codebase)
from league import get_league_id
from player_data import get_players
from score_calculator import POSITION_WEIGHTS, calculate_position_scores
from team import team_stats


def save_json(file_path, data):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        f.write(json.dumps(data))


def ensure_folder_exists(folder_path):
    """Create folder if it doesn't exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def flatten_team_data(team_data, filtered_stats):
    """
    Process team data to create flat player records with position scores.

    Args:
        team_data (dict): Dictionary containing team and player information
        filtered_stats (list): List of stats to include in the output

    Returns:
        list: List of player dictionaries with position scores
    """
    flattened_data = []
    for player in team_data['players']:
        # Extract basic player metadata
        player_info = {
            'team_id': team_data['team_id'],
            'team_name': team_data['team_name'],
            'league_id': team_data['league_id'],
            'league_name': team_data['league_name'],
            'season': team_data['season'],
            'player_id': player['meta_data']['player_id'],
            'player_name': player['meta_data']['player_name'],
            'player_age': player['meta_data']['age'],
            'player_country_code': player['meta_data']['player_country_code']
        }

        # Extract all player statistics
        for category, stats in player['stats'].items():
            if isinstance(stats, dict):
                for key, value in stats.items():
                    player_info[f'{category}_{key}'] = value
            else:
                player_info[category] = stats

        # Add derived stats needed for position scoring
        add_derived_stats(player_info)

        # Calculate position scores
        position_scores = calculate_position_scores(player_info)
        player_info.update(position_scores)

        # Only include players with sufficient playing time (15+ matches)
        # Skip goalkeeper position
        if player_info.get("stats_matches_played", 0) >= 15 and player_info.get("stats_positions", "") != "GK":
            flattened_data.append(player_info)

    return flattened_data


def add_derived_stats(player_info):
    """
    Add derived statistics that may be needed for position scoring.
    These are calculated from existing stats rather than pulled directly from the API.

    Args:
        player_info (dict): Player statistics dictionary to be updated
    """
    # Calculate blocks_sh_blocked if defense_blocks and sh_blocked are available
    if 'defense_blocks' in player_info and 'defense_sh_blocked' in player_info:
        player_info['blocks_sh_blocked'] = player_info['defense_sh_blocked']

    # Calculate touch_opp_box if not already present
    if 'possession_touch_opp_box' in player_info:
        player_info['touch_opp_box'] = player_info['possession_touch_opp_box']

    # Calculate touch_fthird if not already present
    if 'possession_touch_fthird' in player_info:
        player_info['touch_fthird'] = player_info['possession_touch_fthird']

    # Calculate touch_mid_third if not already present
    if 'possession_touch_mid_third' in player_info:
        player_info['touch_mid_third'] = player_info['possession_touch_mid_third']

    # Calculate carries_fthird if not already present
    if 'possession_carries_fthird' in player_info:
        player_info['carries_fthird'] = player_info['possession_carries_fthird']

    # Calculate gca_pass_live_gca if needed
    if 'gca_pass_live_gca' in player_info:
        player_info['gca_pass_live_gca'] = player_info['gca_pass_live_gca']


def add_league_ids():
    """Retrieve league IDs for major European leagues."""
    leagues_needed = [{"country_code": "ENG", "league_name": "Premier League"},
                      {"country_code": "GER", "league_name": "Fußball-Bundesliga"},
                      {"country_code": "FRA", "league_name": "Ligue 1"},
                      {"country_code": "ITA", "league_name": "Serie A"},
                      {"country_code": "ESP", "league_name": "La Liga"}]

    for league in tqdm(leagues_needed, desc="Fetching league IDs"):
        league['league_id'] = get_league_id(league["country_code"], league_name=league["league_name"])
        time.sleep(3)  # Rate limiting

    return leagues_needed


def extract_teams():
    """Extract team data for all leagues."""
    teams = list()
    leagues_needed = add_league_ids()

    for league in tqdm(leagues_needed, desc="Fetching team data"):
        league_data = team_stats(league_id=league['league_id'])
        time.sleep(3)  # Rate limiting

        # Add league information to each team
        for team in league_data:
            team['league_id'] = league['league_id']
            team['league_name'] = league['league_name']

        teams.extend(league_data)

    return teams


def extract_players(filtered_stats):
    """
    Extract player data for all teams across multiple seasons.

    Args:
        filtered_stats (list): List of stats to include in output

    Returns:
        list: List of player dictionaries with stats and position scores
    """
    teams = extract_teams()
    new_teams = list()

    # TODO - remove
    for team in tqdm(teams, desc="Processing teams"):
        for season in ["2023-2024", "2022-2023", "2021-2022"]:
            # Fetch player data for this team and season
            players = get_players(team_id=team['team_id'], league_id=team['league_id'], season_id=season)
            team['players'] = players
            team['season'] = season

            # Process player data
            new_teams.extend(flatten_team_data(team, filtered_stats))
            time.sleep(3)  # Rate limiting

    return new_teams


def calculate_percentiles(df):
    """
    Calculate percentile scores for each position across all players, grouped by season.

    Args:
        df (pandas.DataFrame): DataFrame containing player data with position scores

    Returns:
        pandas.DataFrame: DataFrame with added percentile columns
    """
    # Filter out GK position from POSITION_WEIGHTS
    position_columns = [pos for pos in POSITION_WEIGHTS.keys() if pos != 'GK']

    # Create a copy of the DataFrame to avoid modifying the original
    df_with_percentiles = df.copy()

    # Group by season and calculate percentiles for each position within each season
    for season in df_with_percentiles['season'].unique():
        season_mask = df_with_percentiles['season'] == season

        for position in position_columns:
            # Create a new column for the position percentile
            percentile_column = f"{position}_percentile"

            # Initialize with NaN
            if percentile_column not in df_with_percentiles.columns:
                df_with_percentiles[percentile_column] = float('nan')

            # Calculate percentile rank for this season (reversed so higher values = higher percentiles)
            season_percentiles = df_with_percentiles.loc[season_mask, position].rank(pct=True) * 100

            # Update the values for this season
            df_with_percentiles.loc[season_mask, percentile_column] = season_percentiles

    # Round all percentile columns to 1 decimal place
    percentile_columns = [f"{position}_percentile" for position in position_columns]
    df_with_percentiles[percentile_columns] = df_with_percentiles[percentile_columns].round(1)

    return df_with_percentiles


def calculate_best_positions(df, top_n=3):
    """
    Calculate the best positions for each player based on percentile scores.

    Args:
        df (pandas.DataFrame): DataFrame with player data and position scores
        top_n (int): Number of top positions to include

    Returns:
        pandas.DataFrame: DataFrame with added best positions columns
    """
    # Filter out GK position from POSITION_WEIGHTS
    position_columns = [pos for pos in POSITION_WEIGHTS.keys() if pos != 'GK']

    # Create a copy of the DataFrame
    df_with_best = df.copy()

    # Get score columns (using only percentile scores)
    score_columns = [f"{pos}_percentile" for pos in position_columns]

    # Add best position and score columns
    for i in range(1, top_n + 1):
        df_with_best[f'best_position_{i}'] = None
        df_with_best[f'position_score_{i}'] = None

    # For each player, find their best positions
    for idx, row in df_with_best.iterrows():
        # Create a dictionary of position:score pairs
        position_scores = {pos: row.get(f"{pos}_percentile", 0) for pos in position_columns}

        # Sort positions by score (descending)
        sorted_positions = sorted(position_scores.items(), key=lambda x: x[1], reverse=True)

        # Assign top positions and scores
        for i in range(min(top_n, len(sorted_positions))):
            pos, score = sorted_positions[i]
            df_with_best.at[idx, f'best_position_{i + 1}'] = pos
            df_with_best.at[idx, f'position_score_{i + 1}'] = score

    return df_with_best


def calculate_multi_season_scores(df):
    """
    Calculate weighted position scores across multiple seasons.

    This function takes a DataFrame with position percentile scores for multiple seasons,
    and creates a weighted average that gives more importance to recent seasons.

    Args:
        df (pandas.DataFrame): DataFrame containing player data with position percentile scores

    Returns:
        pandas.DataFrame: DataFrame with added weighted multi-season scores
    """
    # Define season weights - more recent seasons have more weight
    season_weights = {
        "2023-2024": 0.6,  # 60% weight to most recent season
        "2022-2023": 0.3,  # 30% weight to previous season
        "2021-2022": 0.1  # 10% weight to oldest season
    }

    # Filter out GK position from POSITION_WEIGHTS
    position_columns = [pos for pos in POSITION_WEIGHTS.keys() if pos != 'GK']
    percentile_columns = [f"{pos}_percentile" for pos in position_columns]

    # Create a copy of the DataFrame to avoid modifying the original
    df_with_multi = df.copy()

    # Create a column to mark the latest season for each player
    df_with_multi['is_latest_season'] = False

    # Add new columns for weighted scores
    for pos in position_columns:
        df_with_multi[f"{pos}_weighted"] = np.nan

    # Process each player separately
    print("Calculating multi-season weighted scores...")
    for player_id in tqdm(df_with_multi['player_id'].unique()):
        # Get all seasons for this player
        player_mask = df_with_multi['player_id'] == player_id
        player_seasons = df_with_multi.loc[player_mask, 'season'].unique()

        # Skip if player only has one season
        if len(player_seasons) < 2:
            # For single-season players, weighted score = percentile score
            for pos in position_columns:
                df_with_multi.loc[player_mask, f"{pos}_weighted"] = df_with_multi.loc[player_mask, f"{pos}_percentile"]

            # Mark this as the latest season for the player
            latest_season_idx = df_with_multi.loc[player_mask, 'season'].idxmax()
            df_with_multi.loc[latest_season_idx, 'is_latest_season'] = True
            continue

        # Sort seasons by recency
        sorted_seasons = sorted(player_seasons, reverse=True)
        latest_season = sorted_seasons[0]

        # Mark the latest season for this player
        latest_mask = (df_with_multi['player_id'] == player_id) & (df_with_multi['season'] == latest_season)
        df_with_multi.loc[latest_mask, 'is_latest_season'] = True

        # Calculate total weight for normalization
        total_weight = sum(season_weights.get(season, 0) for season in player_seasons)

        # Initialize weighted scores for each position
        weighted_scores = {pos: 0 for pos in position_columns}

        # Calculate weighted average across seasons
        for season in player_seasons:
            season_mask = (df_with_multi['player_id'] == player_id) & (df_with_multi['season'] == season)
            season_weight = season_weights.get(season, 0) / total_weight  # Normalize weight

            # Add weighted contribution from this season
            for pos in position_columns:
                percentile_score = df_with_multi.loc[season_mask, f"{pos}_percentile"].iloc[0]
                if pd.notna(percentile_score):  # Only add if we have a score
                    weighted_scores[pos] += percentile_score * season_weight

        # Update the player's latest season row with the weighted scores
        for pos in position_columns:
            df_with_multi.loc[latest_mask, f"{pos}_weighted"] = round(weighted_scores[pos], 1)

    return df_with_multi


def calculate_best_weighted_positions(df, top_n=3):
    """
    Calculate the best positions for each player based on weighted multi-season scores.

    Args:
        df (pandas.DataFrame): DataFrame with multi-season weighted scores
        top_n (int): Number of top positions to include

    Returns:
        pandas.DataFrame: DataFrame with added best weighted positions columns
    """
    # Filter out GK position from POSITION_WEIGHTS
    position_columns = [pos for pos in POSITION_WEIGHTS.keys() if pos != 'GK']

    # Create a copy of the DataFrame
    df_with_best = df.copy()

    # Add best weighted position and score columns
    for i in range(1, top_n + 1):
        df_with_best[f'best_weighted_position_{i}'] = None
        df_with_best[f'weighted_position_score_{i}'] = None

    # Only process the latest season for each player
    latest_season_mask = df_with_best['is_latest_season'] == True

    # For each player's latest season, find their best positions based on weighted scores
    for idx, row in df_with_best[latest_season_mask].iterrows():
        # Create a dictionary of position:score pairs
        position_scores = {pos: row.get(f"{pos}_weighted", 0) for pos in position_columns}

        # Sort positions by score (descending)
        sorted_positions = sorted(position_scores.items(), key=lambda x: x[1], reverse=True)

        # Assign top positions and scores
        for i in range(min(top_n, len(sorted_positions))):
            pos, score = sorted_positions[i]
            df_with_best.at[idx, f'best_weighted_position_{i + 1}'] = pos
            df_with_best.at[idx, f'weighted_position_score_{i + 1}'] = score

    return df_with_best


def create_excel_with_multiple_sheets(data_frames, sheet_names, output_path):
    """
    Create an Excel file with multiple sheets.

    Args:
        data_frames (list): List of DataFrames to be saved in separate sheets
        sheet_names (list): List of sheet names corresponding to the DataFrames
        output_path (str): Path where the Excel file will be saved
    """
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

    # Write each DataFrame to a different worksheet
    for df, sheet_name in zip(data_frames, sheet_names):
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Close the Pandas Excel writer and output the Excel file
    writer.close()

    print(f"Excel file with multiple sheets saved at {output_path}")


def main():
    """Main function to run the player analysis pipeline."""
    # Gather all stats needed for position scoring
    filtered_stats = []
    for k, v in POSITION_WEIGHTS.items():
        if k != 'GK':  # Skip goalkeeper position
            filtered_stats.extend(v['stats'])

    # Remove duplicates
    filtered_stats = set(filtered_stats)

    # Add essential stats
    filtered_stats.add("stats_matches_played")
    filtered_stats.add("stats_min")
    filtered_stats.add("stats_positions")  # Add positions to filter goalkeepers
    filtered_stats = list(filtered_stats)

    # Add position columns (excluding GK)
    filtered_stats.extend([pos for pos in POSITION_WEIGHTS.keys() if pos != 'GK'])

    # Base columns that should always be included
    base_columns = ['team_id', 'team_name', 'league_id', 'league_name', 'season',
                    'player_id', 'player_name', 'player_age', 'player_country_code']

    # Combine all columns
    filtered_stats = base_columns + filtered_stats

    # Set up output directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_folder_path = os.path.join(current_dir, 'save_data')
    ensure_folder_exists(save_folder_path)

    # Extract player data
    print("Extracting player data...")
    team_data = extract_players(filtered_stats)

    # Save raw data
    save_json(file_path=os.path.join(save_folder_path, 'initial_data.json'), data=team_data)

    # Convert to DataFrame
    df = pd.DataFrame(team_data)

    # Ensure we have all the columns we need
    df = df[list(filtered_stats)]

    # Calculate percentiles for position scores (by season)
    print("Calculating percentile scores...")
    df = calculate_percentiles(df)

    # Add best position analysis based on single season percentiles
    print("Determining best positions for each player (single season)...")
    df = calculate_best_positions(df, top_n=3)

    # Calculate multi-season weighted scores
    print("Calculating multi-season weighted scores...")
    df = calculate_multi_season_scores(df)

    # Add best position analysis based on weighted multi-season scores
    print("Determining best positions based on multi-season weighted scores...")
    df = calculate_best_weighted_positions(df, top_n=3)

    # Update the filtered_stats to include the new columns
    # Filter out GK position
    position_columns = [pos for pos in POSITION_WEIGHTS.keys() if pos != 'GK']
    position_percentile_columns = [f"{position}_percentile" for position in position_columns]
    position_weighted_columns = [f"{position}_weighted" for position in position_columns]
    best_position_columns = [f"best_position_{i}" for i in range(1, 4)]
    position_score_columns = [f"position_score_{i}" for i in range(1, 4)]
    best_weighted_position_columns = [f"best_weighted_position_{i}" for i in range(1, 4)]
    weighted_position_score_columns = [f"weighted_position_score_{i}" for i in range(1, 4)]

    # Add new columns to filtered_stats
    new_columns = (position_percentile_columns + position_weighted_columns +
                   best_position_columns + position_score_columns +
                   best_weighted_position_columns + weighted_position_score_columns +
                   ['is_latest_season'])

    # Create final datasets - Full dataset with all seasons
    full_df = df.copy()

    # Create a version with only the latest season for each player
    latest_season_df = df[df['is_latest_season'] == True].copy()

    # Create two additional DataFrames:
    # 1. Current season data (latest season for each player)
    current_season_df = latest_season_df.copy()

    # 2. Multi-season weighted data
    weighted_season_df = latest_season_df.copy()

    # Select columns for the current season sheet
    current_season_columns = (
            base_columns +
            list(filtered_stats) +
            position_percentile_columns +
            best_position_columns +
            position_score_columns
    )
    current_season_df = current_season_df[current_season_columns]

    # Select columns for the weighted multi-season sheet
    weighted_season_columns = (
            base_columns +
            position_weighted_columns +
            best_weighted_position_columns +
            weighted_position_score_columns
    )
    weighted_season_df = weighted_season_df[weighted_season_columns]

    # Save processed data
    print("Saving processed data...")

    # Save CSV files
    full_df.to_csv(os.path.join(save_folder_path, 'player_data_all_seasons.csv'), index=False)
    latest_season_df.to_csv(os.path.join(save_folder_path, 'player_data_latest_season_with_weighted.csv'), index=False)

    # Save Excel with multiple sheets
    excel_path = os.path.join(save_folder_path, 'player_analysis_data.xlsx')
    create_excel_with_multiple_sheets(
        [current_season_df, weighted_season_df],
        ['Current Season', 'Multi-Season Weighted'],
        excel_path
    )

    # Also save latest season to JSON for web display
    save_json(file_path=os.path.join(save_folder_path, 'player_data_latest_season.json'),
              data=latest_season_df.to_dict(orient='records'))

    # Display sample results
    print(f"\nTotal players across all seasons: {full_df.shape[0]}")
    print(f"Unique players in latest season: {latest_season_df.shape[0]}")

    print("\nSample player from latest season with multi-season weighted scores:")
    sample_columns = base_columns + [
        'best_position_1', 'position_score_1',
        'best_position_2', 'position_score_2',
        'best_position_3', 'position_score_3',
        'best_weighted_position_1', 'weighted_position_score_1',
        'best_weighted_position_2', 'weighted_position_score_2',
        'best_weighted_position_3', 'weighted_position_score_3'
    ]

    # Show a sample player with different single vs multi-season results if possible
    sample_player = latest_season_df[
        latest_season_df['best_position_1'] != latest_season_df['best_weighted_position_1']]
    if len(sample_player) > 0:
        pprint(sample_player[sample_columns].iloc[0].to_dict())
    else:
        pprint(latest_season_df[sample_columns].iloc[0].to_dict())

    print("\nData processing complete!")


if __name__ == '__main__':
    main()