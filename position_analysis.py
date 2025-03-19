import os

import pandas as pd
from tqdm import tqdm

# Correct the import to match your file name
from position_specific_analysis import (
    create_3d_tsne_plot,
    create_top_players_by_position,
    create_player_radar_chart,
    find_specific_versatile_players,
    create_versatility_visualization,
    create_enhanced_stat_leaderboards,
    create_enhanced_per90_stat_leaderboards
)


def infer_position_weights(df):
    """
    Dynamically infer position weights based on available columns.
    """
    # Define specific positions for Amorim's system
    positions = [
        'CB', 'LCB', 'RCB', 'LWB', 'RWB', 'CM', 'CDM', 'CAM', 'LW', 'RW', 'ST'
    ]

    # Identify key statistical columns for different positions
    position_stats = {
        'CB': [
            'defense_tkl', 'defense_int', 'defense_clearances',
            'defense_blocks', 'defense_tkl_def_third'
        ],
        'LCB': [
            'defense_tkl', 'defense_int', 'defense_blocks',
            'passing_pass_prog', 'possession_ttl_carries_prog_dist'
        ],
        'RCB': [
            'defense_tkl', 'defense_int', 'defense_blocks',
            'passing_pass_prog', 'possession_ttl_carries_prog_dist'
        ],
        'LWB': [
            'passing_cross_opp_box', 'possession_ttl_carries_prog_dist',
            'defense_press', 'defense_blocks', 'defense_tkl'
        ],
        'RWB': [
            'passing_cross_opp_box', 'possession_ttl_carries_prog_dist',
            'defense_press', 'defense_blocks', 'defense_tkl'
        ],
        'CM': [
            'passing_pass_prog', 'passing_key_passes',
            'possession_pass_recvd', 'defense_press', 'defense_tkl'
        ],
        'CDM': [
            'defense_tkl', 'defense_int', 'defense_recov',
            'passing_pass_prog', 'possession_pass_recvd'
        ],
        'CAM': [
            'passing_key_passes', 'passing_xa', 'gca_pass_live_sca',
            'possession_pass_recvd', 'gca_pass_live_gca'
        ],
        'LW': [
            'possession_ttl_carries_prog_dist', 'possession_take_on_suc',
            'gca_pass_live_sca', 'passing_cross_opp_box', 'gca_pass_live_gca'
        ],
        'RW': [
            'possession_ttl_carries_prog_dist', 'possession_take_on_suc',
            'gca_pass_live_sca', 'passing_cross_opp_box', 'gca_pass_live_gca'
        ],
        'ST': [
            'stats_gls', 'stats_xg', 'gca_pass_live_gca',
            'stats_sh', 'possession_ttl_carries_prog_dist'
        ]
    }

    # Find available columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Prepare position weights
    POSITION_WEIGHTS = {}

    for pos, stat_keywords in position_stats.items():
        # Find columns that match the keywords
        pos_stats = []
        for keyword in stat_keywords:
            matching_cols = [col for col in numeric_cols if keyword in col.lower()]
            pos_stats.extend(matching_cols)

        # Remove duplicates and limit to top 5 stats
        pos_stats = list(dict.fromkeys(pos_stats))[:5]

        if pos_stats:
            POSITION_WEIGHTS[pos] = {
                'stats': pos_stats,
                'weights': {}
            }

    return POSITION_WEIGHTS


def prepare_dataframe(df):
    """
    Prepare the dataframe for visualization by handling common issues.
    """
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Ensure required columns exist
    if 'player_name' not in df.columns:
        # Try to find a suitable column name
        name_cols = [col for col in df.columns if 'name' in col.lower()]
        if name_cols:
            df = df.rename(columns={name_cols[0]: 'player_name'})
        else:
            df['player_name'] = df.index.astype(str)

    if 'team_name' not in df.columns:
        # Try to find a suitable column name
        team_cols = [col for col in df.columns if 'team' in col.lower()]
        if team_cols:
            df = df.rename(columns={team_cols[0]: 'team_name'})
        else:
            df['team_name'] = 'Unknown'

    # Make sure player_age column exists
    if 'player_age' not in df.columns and 'age' in df.columns:
        df['player_age'] = df['age']

    # Use the latest season if multiple seasons exist
    if 'season' in df.columns:
        latest_season = df['season'].max()
        df['is_latest_season'] = df['season'] == latest_season

    # Add combined stats if needed
    if 'stats_gls' in df.columns and 'stats_gls_and_ast' in df.columns:
        # Derive assists by subtracting goals from total goals+assists
        if 'stats_ast' not in df.columns:
            df['stats_ast'] = df['stats_gls_and_ast'] - df['stats_gls']

    # Fallback: if stats_gls_and_ast doesn't exist, combine existing columns
    if 'stats_gls_and_ast' not in df.columns:
        if 'stats_gls' in df.columns and 'stats_ast' in df.columns:
            df['stats_gls_and_ast'] = df['stats_gls'] + df['stats_ast']

    # Create defense combinations
    if 'defense_tkl' in df.columns and 'defense_int' in df.columns and 'defense_tkl_plus_int' not in df.columns:
        df['defense_tkl_plus_int'] = df['defense_tkl'] + df['defense_int']

    return df


def generate_position_visualizations(file_path, output_dir=None):
    """
    Generate position-specific visualizations from a multi-sheet Excel file.

    Args:
    file_path (str): Path to the Excel file
    output_dir (str, optional): Directory to save visualizations

    Returns:
    dict: Generated figures
    """
    print(f"Loading data from {file_path}...")

    # Load both sheets from the Excel file
    try:
        # Read current season sheet
        df_current = pd.read_excel(file_path, sheet_name='Current Season')

        # Read multi-season weighted sheet
        df_weighted = pd.read_excel(file_path, sheet_name='Multi-Season Weighted')
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

    # Prepare dataframes
    df_current = prepare_dataframe(df_current)
    df_weighted = prepare_dataframe(df_weighted)

    print(f"Successfully loaded data:")
    print(f"Current Season: {len(df_current)} rows, {len(df_current.columns)} columns")
    print(f"Multi-Season Weighted: {len(df_weighted)} rows, {len(df_weighted.columns)} columns")

    # Infer position weights dynamically
    POSITION_WEIGHTS = infer_position_weights(df_current)
    print("Inferred Position Weights:", list(POSITION_WEIGHTS.keys()))

    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(file_path), 'position_visualizations')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {output_dir}")

    # Get all positions from inferred position weights
    positions = ['LCB', 'CB', 'RCB', 'LWB', 'RWB', 'LCM', 'RCM', 'CAM', 'LW', 'RW', 'ST']
    position_groups = {
        'Defense': ['LCB', 'CB', 'RCB', 'LWB', 'RWB'],
        'Midfield': ['LCM', 'RCM', 'CAM'],
        'Forward': ['LW', 'RW', 'ST']
    }

    # Prepare to track generated figures
    figures = {}

    # Create visualization tasks
    visualization_tasks = [
        # 3D TSNE Plot
        {
            'name': 'players_3d_tsne',
            'title': '3D Player Similarity Map (t-SNE)',
            'func': create_3d_tsne_plot,
            'args': {'df_current': df_current, 'df_weighted': df_weighted}
        }
    ]

    # Add top 10 players for each position
    for position in positions:
        visualization_tasks.append({
            'name': f'top_10_{position}',
            'title': f'Top 10 Players for {position}',
            'func': create_top_players_by_position,
            'args': {'df_current': df_current, 'df_weighted': df_weighted, 'position': position, 'top_n': 10}
        })

    # Add player radar charts for top 5 players in each position
    for position in positions:
        visualization_tasks.append({
            'name': f'radar_{position}',
            'title': f'Radar Chart - Top 5 Players for {position}',
            'func': create_player_radar_chart,
            'args': {'df_current': df_current, 'df_weighted': df_weighted, 'position': position, 'top_n': 5}
        })

    # Add versatility analysis
    versatility_df = find_specific_versatile_players(df_current, df_weighted)
    visualization_tasks.append({
        'name': 'versatility_top20',
        'title': 'Top 20 Most Versatile Players',
        'func': create_versatility_visualization,
        'args': {'df_current': df_current, 'df_weighted': df_weighted, 'top_n': 20}
    })

    # Add stat leaderboards including assists
    visualization_tasks.append({
        'name': 'enhanced_stat_leaderboards',
        'title': 'Key Statistic Leaderboards (with Assists)',
        'func': create_enhanced_stat_leaderboards,
        'args': {'df_current': df_current, 'df_weighted': df_weighted, 'top_n': 10}
    })

    # Add per90 leaderboards including assists/90
    visualization_tasks.append({
        'name': 'enhanced_per90_stat_leaderboards',
        'title': 'Per 90 Minutes Stat Leaderboards (with Assists/90)',
        'func': create_enhanced_per90_stat_leaderboards,
        'args': {'df_current': df_current, 'df_weighted': df_weighted, 'top_n': 10, 'min_minutes': 900}
    })

    # Generate and save visualizations
    for task in tqdm(visualization_tasks, desc="Generating visualizations"):
        try:
            print(f"Creating {task['name']}...")

            # Run the visualization function
            fig = task['func'](**task['args'])

            # Save as HTML
            html_path = os.path.join(output_dir, f"{task['name']}.html")
            fig.write_html(html_path)

            # Try to save as PNG (requires additional dependencies)
            try:
                png_path = os.path.join(output_dir, f"{task['name']}.png")
                fig.write_image(png_path, width=1200, height=800)
            except Exception as e:
                print(f"Warning: Could not save PNG for {task['name']}: {e}")

            # Store the figure
            figures[task['name']] = fig

            print(f"✓ Saved {task['name']}")
        except Exception as e:
            print(f"✗ Error generating {task['name']}: {e}")
            import traceback
            print(traceback.format_exc())

    # Create a dashboard HTML that links to all visualizations
    try:
        create_dashboard(visualization_tasks, output_dir, position_groups)
    except Exception as e:
        print(f"✗ Error creating dashboard: {e}")
        import traceback
        print(traceback.format_exc())

    print(f"\nSuccessfully generated {len(figures)} visualizations in {output_dir}")
    return figures


def create_dashboard(visualization_tasks, output_dir, position_groups):
    """
    Create an HTML dashboard that links to all visualizations.
    """
    # Create HTML content
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Position-Specific Football Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }
        .container {
            width: 90%;
            margin: auto;
            overflow: hidden;
            padding: 20px;
        }
        .header {
            background: #1e3a8a;
            color: #fff;
            padding-top: 30px;
            min-height: 70px;
            border-bottom: #38bdf8 3px solid;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            text-align: center;
        }
        .visualization-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            grid-gap: 20px;
            margin-top: 20px;
        }
        .visualization-card {
            background: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .visualization-card:hover {
            transform: scale(1.05);
        }
        .visualization-card a {
            text-decoration: none;
            color: #0369a1;
            font-weight: bold;
        }
        .category-section {
            margin-top: 30px;
            padding: 15px;
            background: #e9e9e9;
            border-radius: 5px;
        }
        .category-section h2 {
            border-bottom: 2px solid #0369a1;
            padding-bottom: 10px;
            color: #333;
        }
        .position-section {
            margin-top: 20px;
            background: #f0f9ff;
            border-radius: 5px;
            padding: 10px;
        }
        .position-section h3 {
            color: #0369a1;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Position-Specific Football Analysis</h1>
    </div>

    <div class="container">
    """

    # Add Position Embeddings Section with 3D TSNE plot
    html += """
        <div class="category-section">
            <h2>Position Embeddings (t-SNE)</h2>
            <div class="visualization-grid">
    """

    for task in visualization_tasks:
        if task['name'] == 'players_3d_tsne':
            html += f"""
                    <div class='visualization-card'>
                        <a href="{task['name']}.html" target="_blank">{task['title']}</a>
                    </div>
                """

    html += """
                </div>
            </div>
        """

    # Add Position-Specific Analysis Sections
    for group_name, group_positions in position_groups.items():
        html += f"""
        <div class="category-section">
            <h2>{group_name} Position Analysis</h2>
        """

        for position in group_positions:
            html += f"""
            <div class="position-section">
                <h3>{position}</h3>
                <div class="visualization-grid">
            """

            # Add top 10 players for this position
            for task in visualization_tasks:
                if task['name'] == f'top_10_{position}':
                    html += f"""
                    <div class='visualization-card'>
                        <a href="{task['name']}.html" target="_blank">{task['title']}</a>
                    </div>
                    """

            # Add radar chart for this position
            for task in visualization_tasks:
                if task['name'] == f'radar_{position}':
                    html += f"""
                    <div class='visualization-card'>
                        <a href="{task['name']}.html" target="_blank">{task['title']}</a>
                    </div>
                    """

            html += """
                </div>
            </div>
            """

        html += """
        </div>
        """

    # Add Versatility Analysis Section
    html += """
        <div class="category-section">
            <h2>Versatility Analysis</h2>
            <div class="visualization-grid">
    """

    for task in visualization_tasks:
        if 'versatility' in task['name']:
            html += f"""
            <div class='visualization-card'>
                <a href="{task['name']}.html" target="_blank">{task['title']}</a>
            </div>
            """

    html += """
            </div>
        </div>
    """

    # Add Statistical Leaders Section
    html += """
        <div class="category-section">
            <h2>Statistical Leaders</h2>
            <div class="visualization-grid">
    """

    for task in visualization_tasks:
        if 'stat_leaderboards' in task['name']:
            html += f"""
            <div class='visualization-card'>
                <a href="{task['name']}.html" target="_blank">{task['title']}</a>
            </div>
            """

    html += """
            </div>
        </div>
    """

    # Close HTML tags
    html += """
    </div>
</body>
</html>
"""

    # Write to file
    dashboard_path = os.path.join(output_dir, "position_dashboard.html")
    with open(dashboard_path, 'w') as f:
        f.write(html)

    print(f"✓ Dashboard created at {dashboard_path}")


if __name__ == '__main__':
    # If run as a script, get the file path from command line arguments
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate Position-Specific Football Analysis Visualizations')

    # Add arguments
    parser.add_argument('file_path', type=str,
                        help='Path to the Excel file containing football player data')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Directory to save visualizations (optional)')
    parser.add_argument('-p', '--positions', nargs='+',
                        default=['LCB', 'CB', 'RCB', 'LWB', 'RWB', 'LCM', 'RCM', 'CAM', 'LW', 'RW', 'ST'],
                        help='Specific positions to analyze (optional)')
    parser.add_argument('-n', '--top_n', type=int, default=10,
                        help='Number of top players to show in leaderboards (default: 10)')
    parser.add_argument('-m', '--min_minutes', type=int, default=900,
                        help='Minimum minutes played for per 90 stats (default: 900)')

    # Parse arguments
    args = parser.parse_args()

    try:
        # Generate visualizations
        figures = generate_position_visualizations(
            args.file_path,
            output_dir=args.output_dir
        )

        print(f"\nSuccessfully generated {len(figures)} visualizations.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
