import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def map_player_position(position_str):
    """
    Map player's position string to specific position categories.

    Args:
    position_str (str): A string representing player's position characteristics

    Returns:
    list: A list of position categories the player can play
    """
    # Specific position mapping for detailed positions
    position_categories = {
        'Defense': ['DF', 'LCB', 'CB', 'RCB', 'LWB', 'RWB'],
        'Midfield': ['MF', 'LCM', 'RCM', 'CAM'],
        'Forward': ['FW', 'LW', 'RW', 'ST']
    }

    # Detailed position mapping
    detailed_positions = ['LCB', 'CB', 'RCB', 'LWB', 'RWB',
                          'LCM', 'RCM', 'CAM',
                          'LW', 'RW', 'ST']

    # Categorize player's positions
    player_positions = []

    # Check for broad categories
    for category, codes in position_categories.items():
        if any(code in position_str for code in codes):
            player_positions.append(category)

    # Check for specific positions
    specific_player_positions = [
        pos for pos in detailed_positions
        if pos in position_str or
           (pos == 'LCM' and 'CM' in position_str) or
           (pos == 'RCM' and 'CM' in position_str)
    ]

    # Combine and remove duplicates
    player_positions.extend(specific_player_positions)
    player_positions = list(set(player_positions))

    return player_positions


def filter_players_by_position(df, target_position):
    """
    Filter players based on their position characteristics.

    Args:
    df (pd.DataFrame): Input dataframe
    target_position (str): Target position to filter

    Returns:
    pd.DataFrame: Filtered dataframe
    """
    # Mapping of broad positions to specific positions
    position_mapping = {
        'Defense': ['LCB', 'CB', 'RCB', 'LWB', 'RWB'],
        'Midfield': ['LCM', 'RCM', 'CAM'],
        'Forward': ['LW', 'RW', 'ST']
    }

    # Determine the position group
    target_group = None
    for group, positions in position_mapping.items():
        if target_position in positions:
            target_group = group
            break

    if not target_group:
        return df  # Return all players if no specific group found

    # Add a column with position categories if not exists
    if 'position_categories' not in df.columns:
        df['position_categories'] = df['stats_positions'].apply(map_player_position)

    # Filter players
    filtered_df = df[
        df['position_categories'].apply(
            lambda x: target_group in x or target_position in x
        )
    ]

    return filtered_df


def get_position_stats_for_tsne(df):
    """
    Select appropriate statistical columns for t-SNE embedding.

    Args:
    df (pd.DataFrame): Input dataframe

    Returns:
    list: List of statistical columns for embedding
    """
    # Comprehensive list of statistical columns for embedding
    embedding_cols = [
        'gca_ttl_sca', 'possession_ttl_carries_prog_dist',
        'touch_opp_box', 'defense_tkl_plus_int',
        'passing_pct_pass_cmp', 'defense_int',
        'gca_pass_live_gca', 'stats_gls_and_ast',
        'shooting_sh', 'defense_clearances',
        'passing_key_passes', 'gca_pass_live_sca',
        'touch_mid_third', 'defense_blocks',
        'possession_pass_recvd', 'gca_take_on_sca',
        'misc_air_dual_won', 'passing_pass_prog',
        'misc_pct_air_dual_won', 'stats_gls',
        'carries_fthird', 'defense_def_error',
        'possession_take_on_suc', 'defense_tkl_def_third',
        'passing_cross_opp_box', 'defense_tkl',
        'passing_xa', 'shooting_gls_xg_diff',
        'defense_tkl_mid_third', 'misc_ball_recov'
    ]

    # Ensure all columns exist in the dataframe
    valid_cols = [col for col in embedding_cols if col in df.columns]

    return valid_cols


def create_3d_tsne_plot(df_current, df_weighted):
    """
    Create a 3D t-SNE visualization with all players using multi-season weighted positions.

    Args:
    df_current (pd.DataFrame): Current season dataframe
    df_weighted (pd.DataFrame): Multi-season weighted dataframe

    Returns:
    plotly.graph_objs._figure.Figure: TSNE visualization
    """
    # Merge current season and weighted dataframes
    merged_df = pd.merge(
        df_current,
        df_weighted[['player_id', 'best_weighted_position_1', 'best_weighted_position_2', 'best_weighted_position_3']],
        on='player_id',
        how='left'
    )

    # Prepare embedding columns
    embedding_cols = get_position_stats_for_tsne(merged_df)

    # Fill missing values with median
    for col in embedding_cols:
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())

    # Extract feature matrix using statistical columns
    X = merged_df[embedding_cols].values

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply t-SNE with 3 components
    reducer = TSNE(n_components=3, random_state=42, perplexity=min(50, len(X_scaled) - 1))
    embedding = reducer.fit_transform(X_scaled)

    # Create a DataFrame with the embedding
    embed_df = pd.DataFrame(embedding, columns=['x', 'y', 'z'])

    # Add player information
    embed_df['player_name'] = merged_df['player_name'].values
    embed_df['position'] = merged_df['best_weighted_position_1'].values
    embed_df['team'] = merged_df['team_name'].values
    embed_df['age'] = merged_df['player_age'].values

    # Create 3D scatter plot with a predefined color palette
    position_colors = {
        'LCB': '#1f77b4',  # Blue
        'CB': '#ff7f0e',  # Orange
        'RCB': '#2ca02c',  # Green
        'LWB': '#d62728',  # Red
        'RWB': '#9467bd',  # Purple
        'LCM': '#8c564b',  # Brown
        'RCM': '#e377c2',  # Pink
        'CAM': '#7f7f7f',  # Gray
        'LW': '#bcbd22',  # Olive
        'RW': '#17becf',  # Cyan
        'ST': '#ff9896'  # Light Red
    }

    # Default color for unknown positions
    default_color = '#1f77b4'

    # Create the figure
    fig = px.scatter_3d(
        embed_df, x='x', y='y', z='z',
        color='position',
        color_discrete_map=position_colors,
        hover_name='player_name',
        hover_data={
            'x': False,
            'y': False,
            'z': False,
            'team': True,
            'age': True,
            'position': True,
        },
        title='3D Player Similarity Map (t-SNE)',
        opacity=0.8
    )

    # Improve layout
    fig.update_layout(
        template="plotly_white",
        legend_title_text='Primary Position',
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3",
            aspectmode='cube'  # Equal aspect ratio
        ),
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=0, r=0, t=50, b=0),
        height=800,
        width=1000
    )

    # Improve marker appearance
    fig.update_traces(
        marker=dict(
            size=5,
            line=dict(width=0.5, color='DarkSlateGrey')
        )
    )

    return fig


def create_top_players_by_position(df_current, df_weighted, position, top_n=10):
    """
    Create a visualization showing top N players for a specific position
    based on multi-season weighted scores.

    Args:
    df_current (pd.DataFrame): Current season dataframe
    df_weighted (pd.DataFrame): Multi-season weighted dataframe
    position (str): Target position (e.g., 'LCM', 'LWB')
    top_n (int): Number of top players to show

    Returns:
    plotly.graph_objs._figure.Figure: Top players visualization
    """
    # Merge current season and weighted dataframes
    merged_df = pd.merge(
        df_current,
        df_weighted[['player_id', 'LCB_weighted', 'CB_weighted', 'RCB_weighted',
                     'LWB_weighted', 'RWB_weighted', 'LCM_weighted', 'RCM_weighted',
                     'CAM_weighted', 'LW_weighted', 'RW_weighted', 'ST_weighted']],
        on='player_id',
        how='left'
    )

    # Mapping of broad position groups
    position_groups = {
        'Defense': ['LCB', 'CB', 'RCB', 'LWB', 'RWB'],
        'Midfield': ['LCM', 'RCM', 'CAM'],
        'Forward': ['LW', 'RW', 'ST']
    }

    # Determine the group for the target position
    target_group = next((group for group, positions in position_groups.items() if position in positions), None)

    if not target_group:
        return go.Figure().update_layout(title=f"No group found for position: {position}")

    # Filter players by position group
    filtered_df = filter_players_by_position(merged_df, target_group)

    # Find column with weighted score for this position
    weighted_col = f"{position}_weighted"

    if weighted_col not in filtered_df.columns:
        return go.Figure().update_layout(title=f"No weighted column found for position: {position}")

    # Get top N players for this position based on weighted score
    top_players = filtered_df.nlargest(top_n, weighted_col).copy()

    # Add rank column
    top_players['rank'] = range(1, len(top_players) + 1)

    # Color palette with distinct colors for different positions
    position_colors = {
        'LCB': 'rgba(31, 119, 180, 0.8)',  # Blue
        'CB': 'rgba(255, 127, 14, 0.8)',  # Orange
        'RCB': 'rgba(44, 160, 44, 0.8)',  # Green
        'LWB': 'rgba(214, 39, 40, 0.8)',  # Red
        'RWB': 'rgba(148, 103, 189, 0.8)',  # Purple
        'LCM': 'rgba(140, 86, 75, 0.8)',  # Brown
        'RCM': 'rgba(227, 119, 194, 0.8)',  # Pink
        'CAM': 'rgba(127, 127, 127, 0.8)',  # Gray
        'LW': 'rgba(188, 189, 34, 0.8)',  # Olive
        'RW': 'rgba(23, 190, 207, 0.8)',  # Cyan
        'ST': 'rgba(255, 152, 150, 0.8)'  # Light Red
    }

    # Get color for this position, default to blue if not found
    bar_color = position_colors.get(position, 'rgba(31, 119, 180, 0.8)')

    # Create horizontal bar chart
    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        x=top_players[weighted_col],
        y=top_players['player_name'],
        orientation='h',
        text=[f"#{rank}" for rank in top_players['rank']],
        textposition='inside',
        marker=dict(
            color=bar_color,
            line=dict(color=bar_color.replace('0.8', '1.0'), width=2)
        ),
        hovertext=[
            f"#{rank} {name} ({team})<br>{position} Score: {score:.1f}"
            for rank, name, team, score in
            zip(top_players['rank'], top_players['player_name'],
                top_players['team_name'], top_players[weighted_col])
        ],
        hoverinfo='text'
    ))

    # Update layout
    fig.update_layout(
        template="plotly_white",
        title=f"Top {top_n} Players for {position}",
        xaxis_title=f"{position} Weighted Score",
        yaxis_title="Player",
        yaxis=dict(autorange="reversed"),  # Highest values at top
        font=dict(family="Arial, sans-serif", size=12),
        height=min(600, 200 + 30 * top_n),
        margin=dict(l=150, r=30, t=80, b=50)
    )

    return fig


def create_player_radar_chart(df_current, df_weighted, position, top_n=5):
    """
    Create a radar chart showing top N players for a specific position,
    using percentile scores from current season data.

    Args:
    df_current (pd.DataFrame): Current season dataframe
    df_weighted (pd.DataFrame): Multi-season weighted dataframe
    position (str): Target position (e.g., 'LCM', 'LWB')
    top_n (int): Number of top players to show

    Returns:
    plotly.graph_objs._figure.Figure: Radar chart visualization
    """
    # Detailed mapping of position groups and their key statistics
    position_stats_mapping = {
        'Defense': {
            'positions': ['LCB', 'CB', 'RCB', 'LWB', 'RWB'],
            'stats': [
                'defense_tkl', 'defense_int', 'defense_clearances',
                'defense_blocks', 'possession_ttl_carries_prog_dist'
            ]
        },
        'Midfield': {
            'positions': ['LCM', 'RCM', 'CAM'],
            'stats': [
                'passing_pass_prog', 'passing_key_passes',
                'possession_pass_recvd', 'gca_pass_live_sca', 'defense_tkl'
            ]
        },
        'Forward': {
            'positions': ['LW', 'RW', 'ST'],
            'stats': [
                'stats_gls', 'gca_ttl_sca', 'possession_take_on_suc',
                'touch_opp_box', 'passing_cross_opp_box'
            ]
        }
    }

    # Determine the position group
    target_group = next((group for group, group_data in position_stats_mapping.items()
                         if position in group_data['positions']), None)

    if not target_group:
        return go.Figure().update_layout(title=f"No group found for position: {position}")

    # Prepare the merged dataframe
    merged_df = pd.merge(
        df_current,
        df_weighted[['player_id', 'best_weighted_position_1', 'best_weighted_position_2', 'best_weighted_position_3']],
        on='player_id',
        how='left'
    )

    # Get stats for the target group
    stats_to_plot = position_stats_mapping[target_group]['stats']

    # Create percentile columns if they don't exist
    for stat in stats_to_plot:
        percentile_col = f"{stat}_percentile"
        if percentile_col not in merged_df.columns:
            # Calculate percentile for the stat
            merged_df[percentile_col] = merged_df[stat].rank(pct=True) * 100

    # Prepare percentile columns
    percentile_columns = [f"{stat}_percentile" for stat in stats_to_plot]

    # Create position-specific percentile column if not exists
    position_percentile_col = f"{position}_percentile"
    if position_percentile_col not in merged_df.columns:
        # Use average of available percentile columns
        merged_df[position_percentile_col] = merged_df[percentile_columns].mean(axis=1)

    # Filter and select top players
    top_players = merged_df.nlargest(top_n, position_percentile_col)

    # Improve labels
    stat_labels = []
    for stat in stats_to_plot:
        parts = stat.split('_')
        if len(parts) > 1:
            # Remove common prefixes
            if parts[0] in ['stats', 'passing', 'defense', 'possession', 'gca']:
                parts = parts[1:]
        stat_labels.append(' '.join(parts).title())

    # Create radar chart
    fig = go.Figure()

    # Distinct color palette for 5 players
    color_palette = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd'  # Purple
    ]

    # Add a trace for each player
    for i, (_, player) in enumerate(top_players.iterrows()):
        # Get percentile values
        stat_values = []
        valid_labels = []
        for stat, label in zip(stats_to_plot, stat_labels):
            percentile_col = f"{stat}_percentile"
            stat_values.append(player[percentile_col])
            valid_labels.append(label)

        # Add radar trace
        fig.add_trace(go.Scatterpolar(
            r=stat_values,
            theta=valid_labels,
            fill='toself',
            name=f"{player['player_name']} ({player['team_name']})",
            line_color=color_palette[i]
        ))

    # Update layout
    fig.update_layout(
        template="plotly_white",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title=f"Top {top_n} Players for {position} (Percentile Scores)",
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=80, r=80, t=100, b=100),
        height=600,
        width=700
    )

    return fig


def find_specific_versatile_players(df_current, df_weighted, min_percentile=70, positions=None):
    """
    Find players that can play multiple specific positions effectively.

    Args:
    df_current (pd.DataFrame): Current season dataframe with percentile columns
    df_weighted (pd.DataFrame): Multi-season weighted dataframe
    min_percentile (float): Minimum percentile threshold for considering a position
    positions (list): List of positions to check for versatility

    Returns:
    pd.DataFrame: DataFrame of versatile players
    """
    # Merge current season and weighted dataframes
    merged_df = pd.merge(
        df_current,
        df_weighted[['player_id', 'best_weighted_position_1', 'best_weighted_position_2', 'best_weighted_position_3']],
        on='player_id',
        how='left'
    )

    # Default positions if not provided
    if positions is None:
        positions = ['LCB', 'CB', 'RCB', 'LWB', 'RWB', 'LCM', 'RCM', 'CAM', 'LW', 'RW', 'ST']

    # Prepare result data
    result_data = []

    # Percentile columns
    percentile_cols = [f"{pos}_percentile" for pos in positions]
    valid_percentile_cols = [col for col in percentile_cols if col in merged_df.columns]

    # For each player, check which positions they can play
    for _, player in merged_df.iterrows():
        viable_positions = []
        position_scores = {}

        # Check each position
        for pos in positions:
            pos_percentile_col = f"{pos}_percentile"

            # Check if percentile column exists and has a valid value
            if pos_percentile_col in merged_df.columns and not pd.isna(player.get(pos_percentile_col, np.nan)):
                pos_percentile = player[pos_percentile_col]

                # Check if player meets versatility criteria
                if pos_percentile >= min_percentile:
                    viable_positions.append(pos)
                    position_scores[pos] = pos_percentile

        # If player is versatile (can play multiple positions), add to results
        if len(viable_positions) > 1:  # Must be able to play at least 2 positions
            result_data.append({
                'player_name': player.get('player_name', 'Unknown'),
                'team': player.get('team_name', 'Unknown'),
                'main_position': player.get('best_weighted_position_1', 'Unknown'),
                'age': player.get('player_age', None),
                'versatility_count': len(viable_positions),
                'versatile_positions': ', '.join(viable_positions),
                'position_scores': position_scores
            })

    # Convert to DataFrame
    versatility_df = pd.DataFrame(result_data)

    # Sort by versatility count (number of positions they can play)
    if not versatility_df.empty:
        versatility_df = versatility_df.sort_values('versatility_count', ascending=False)

    return versatility_df


def create_versatility_visualization(df_current, df_weighted, top_n=20):
    """
    Create a visualization showing the most versatile players.

    Args:
    df_current (pd.DataFrame): Current season dataframe
    df_weighted (pd.DataFrame): Multi-season weighted dataframe
    top_n (int): Number of top versatile players to show

    Returns:
    plotly.graph_objs._figure.Figure: Versatility visualization
    """
    # Find versatile players using the previously defined function
    versatility_df = find_specific_versatile_players(
        df_current,
        df_weighted,
        min_percentile=70,
        positions=['LCB', 'CB', 'RCB', 'LWB', 'RWB', 'LCM', 'RCM', 'CAM', 'LW', 'RW', 'ST']
    )

    # Ensure the required columns exist
    if versatility_df.empty:
        return go.Figure().update_layout(title="No versatile players found")

    # Get top players
    top_players = versatility_df.head(top_n)

    # Create horizontal bar chart
    fig = go.Figure()

    # Color gradient for versatility count
    color_scale = [
        [0, 'rgba(214, 39, 40, 0.7)'],  # Red for low versatility
        [0.5, 'rgba(255, 127, 14, 0.7)'],  # Orange for medium versatility
        [1, 'rgba(44, 160, 44, 0.7)']  # Green for high versatility
    ]

    # Add bars
    fig.add_trace(go.Bar(
        x=top_players['versatility_count'],
        y=top_players['player_name'],
        orientation='h',
        text=top_players['versatile_positions'],
        textposition='auto',
        marker=dict(
            color=top_players['versatility_count'],
            colorscale=color_scale,
            colorbar=dict(title="Number of Positions")
        ),
        hovertext=[
            f"{name} ({team})<br>Versatile Positions: {positions}<br>Detailed Scores:<br>" +
            "<br>".join([f"{pos}: {score:.1f}" for pos, score in scores.items()])
            for name, team, positions, scores in
            zip(top_players['player_name'],
                top_players['team'],
                top_players['versatile_positions'],
                top_players['position_scores'])
        ],
        hoverinfo='text'
    ))

    # Update layout
    fig.update_layout(
        template="plotly_white",
        title=f"Top {top_n} Most Versatile Players",
        xaxis_title="Number of Positions",
        yaxis_title="Player",
        yaxis=dict(autorange="reversed"),  # Highest values at top
        font=dict(family="Arial, sans-serif", size=12),
        height=min(800, 200 + 20 * top_n),  # Dynamic height based on number of players
        margin=dict(l=150, r=30, t=80, b=50)
    )

    return fig


def create_enhanced_stat_leaderboards(df_current, df_weighted, stats_to_show=None, top_n=10):
    """
    Create leaderboards for key statistics including assists.

    Args:
    df_current (pd.DataFrame): Current season dataframe
    df_weighted (pd.DataFrame): Multi-season weighted dataframe
    stats_to_show (list): List of statistics to display
    top_n (int): Number of top players to show for each statistic

    Returns:
    plotly.graph_objs._figure.Figure: Statistical leaderboards visualization
    """
    # Default stats to show if none provided
    if stats_to_show is None:
        stats_to_show = [
            {'name': 'Goals', 'col': 'stats_gls', 'min_matches': 15},
            {'name': 'Assists', 'col': 'stats_ast', 'min_matches': 15},
            {'name': 'Goal+Assist', 'col': 'stats_gls_and_ast', 'min_matches': 15},
            {'name': 'Shot-Creating Actions', 'col': 'gca_ttl_sca', 'min_matches': 15},
            {'name': 'Progressive Passes', 'col': 'passing_pass_prog', 'min_matches': 15},
            {'name': 'Key Passes', 'col': 'passing_key_passes', 'min_matches': 15},
            {'name': 'Successful Take-Ons', 'col': 'possession_take_on_suc', 'min_matches': 15}
        ]

    # Merge current season and weighted dataframes
    merged_df = pd.merge(
        df_current,
        df_weighted[['player_id', 'best_weighted_position_1']],
        on='player_id',
        how='left'
    )

    # Create subplots for each stat
    n_stats = len(stats_to_show)
    n_cols = min(2, n_stats)
    n_rows = (n_stats + n_cols - 1) // n_cols  # Ceiling division

    # Calculate subplot sizes
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[stat['name'] for stat in stats_to_show]
    )

    # Color palette
    color_palette = [
        'rgba(31, 119, 180, 0.7)',  # Blue
        'rgba(255, 127, 14, 0.7)',  # Orange
        'rgba(44, 160, 44, 0.7)',  # Green
        'rgba(214, 39, 40, 0.7)',  # Red
        'rgba(148, 103, 189, 0.7)'  # Purple
    ]

    # Add each stat leaderboard
    for i, stat_info in enumerate(stats_to_show):
        stat_name = stat_info['name']
        stat_col = stat_info['col']
        min_matches = stat_info.get('min_matches', 15)

        # Skip if stat column not available
        if stat_col not in merged_df.columns:
            print(f"Warning: {stat_col} not found in dataframe, skipping {stat_name}")
            continue

        # Filter by minimum matches
        if 'stats_matches_played' in merged_df.columns:
            qualified_df = merged_df[merged_df['stats_matches_played'] >= min_matches]
        else:
            qualified_df = merged_df

        # Sort by the stat and get top players
        top_players = qualified_df.sort_values(stat_col, ascending=False).head(top_n)

        # Calculate row and column for this subplot
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        # Create horizontal bar chart
        fig.add_trace(
            go.Bar(
                x=top_players[stat_col],
                y=top_players['player_name'],
                orientation='h',
                text=top_players[stat_col].round(1),
                textposition='outside',
                marker=dict(
                    color=color_palette[i % len(color_palette)],
                    line=dict(width=1.5, color='darkgray')
                ),
                hovertext=[
                    f"{player} ({team})<br>Position: {position}<br>Value: {value:.1f}"
                    for player, team, position, value in
                    zip(top_players['player_name'],
                        top_players['team_name'],
                        top_players['best_weighted_position_1'],
                        top_players[stat_col])
                ],
                hoverinfo='text',
                showlegend=False
            ),
            row=row, col=col
        )

        # Customize axes
        fig.update_xaxes(title=stat_name, row=row, col=col)
        fig.update_yaxes(autorange="reversed", row=row, col=col)  # Highest values at top

    # Update layout
    fig.update_layout(
        template="plotly_white",
        height=400 * n_rows,
        width=600 * n_cols,
        title_text="Key Statistic Leaderboards",
        margin=dict(l=100, r=30, t=80, b=30),
        font=dict(family="Arial, sans-serif", size=12)
    )

    return fig


def create_enhanced_per90_stat_leaderboards(df_current, df_weighted, stats_to_show=None, top_n=10, min_minutes=900):
    """
    Create leaderboards for key statistics per 90 minutes including assists/90.

    Args:
    df_current (pd.DataFrame): Current season dataframe
    df_weighted (pd.DataFrame): Multi-season weighted dataframe
    stats_to_show (list): List of statistics to display
    top_n (int): Number of top players to show for each statistic
    min_minutes (int): Minimum minutes played to be considered

    Returns:
    plotly.graph_objs._figure.Figure: Per 90 minutes statistical leaderboards
    """
    # Default stats to show if none provided
    if stats_to_show is None:
        stats_to_show = [
            {'name': 'Goals/90', 'col': 'stats_gls'},
            {'name': 'Assists/90', 'col': 'stats_ast'},
            {'name': 'G+A/90', 'col': 'stats_gls_and_ast'},
            {'name': 'SCA/90', 'col': 'gca_ttl_sca'},
            {'name': 'Prog Passes/90', 'col': 'passing_pass_prog'},
            {'name': 'Key Passes/90', 'col': 'passing_key_passes'},
            {'name': 'Tkl+Int/90', 'col': 'defense_tkl_plus_int'},
            {'name': 'Take-Ons/90', 'col': 'possession_take_on_suc'}
        ]

    # Merge current season and weighted dataframes
    merged_df = pd.merge(
        df_current,
        df_weighted[['player_id', 'best_weighted_position_1']],
        on='player_id',
        how='left'
    )

    # Filter by minimum minutes
    if 'stats_min' in merged_df.columns:
        merged_df = merged_df[merged_df['stats_min'] >= min_minutes].copy()

    # Calculate per 90 stats
    for stat_info in stats_to_show:
        stat_col = stat_info['col']
        if stat_col in merged_df.columns and 'stats_min' in merged_df.columns:
            merged_df[f"{stat_col}_per90"] = merged_df[stat_col] / (merged_df['stats_min'] / 90)

    # Create subplots for each stat
    n_stats = len(stats_to_show)
    n_cols = min(2, n_stats)
    n_rows = (n_stats + n_cols - 1) // n_cols  # Ceiling division

    # Calculate subplot sizes
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[stat['name'] for stat in stats_to_show]
    )

    # Color palette
    color_palette = [
        'rgba(31, 119, 180, 0.7)',  # Blue
        'rgba(255, 127, 14, 0.7)',  # Orange
        'rgba(44, 160, 44, 0.7)',  # Green
        'rgba(214, 39, 40, 0.7)',  # Red
        'rgba(148, 103, 189, 0.7)',  # Purple
        'rgba(140, 86, 75, 0.7)',  # Brown
        'rgba(227, 119, 194, 0.7)',  # Pink
        'rgba(127, 127, 127, 0.7)'  # Gray
    ]

    # Add each stat leaderboard
    for i, stat_info in enumerate(stats_to_show):
        stat_name = stat_info['name']
        stat_col = stat_info['col']
        per90_col = f"{stat_col}_per90"

        # Skip if per90 stat column not available
        if per90_col not in merged_df.columns:
            print(f"Warning: {per90_col} not found in dataframe, skipping {stat_name}")
            continue

        # Sort by the per90 stat and get top players
        top_players = merged_df.sort_values(per90_col, ascending=False).head(top_n)

        # Calculate row and column for this subplot
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        # Create horizontal bar chart
        fig.add_trace(
            go.Bar(
                x=top_players[per90_col],
                y=top_players['player_name'],
                orientation='h',
                text=top_players[per90_col].round(2),
                textposition='outside',
                marker=dict(
                    color=color_palette[i % len(color_palette)],
                    line=dict(width=1.5, color='darkgray')
                ),
                hovertext=[
                    f"{player} ({team})<br>Position: {position}<br>Value/90: {value:.2f}<br>Total Minutes: {mins}"
                    for player, team, position, value, mins in
                    zip(top_players['player_name'],
                        top_players['team_name'],
                        top_players['best_weighted_position_1'],
                        top_players[per90_col],
                        top_players['stats_min'])
                ],
                hoverinfo='text',
                showlegend=False
            ),
            row=row, col=col
        )

        # Customize axes
        fig.update_xaxes(title=stat_name, row=row, col=col)
        fig.update_yaxes(autorange="reversed", row=row, col=col)  # Highest values at top

    # Update layout
    fig.update_layout(
        template="plotly_white",
        height=400 * n_rows,
        width=600 * n_cols,
        title_text=f"Per 90 Minutes Stat Leaderboards (min. {min_minutes} minutes)",
        margin=dict(l=100, r=30, t=80, b=30),
        font=dict(family="Arial, sans-serif", size=12)
    )

    return fig
