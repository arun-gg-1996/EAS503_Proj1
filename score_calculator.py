from pprint import pprint

# Define weights for all positions in Amorim's 5-2-3 system
# GK position is included but will be filtered out in the main code
POSITION_WEIGHTS = {
    # Center-Backs (asymmetric roles)
    'LCB': {
        'weights': {
            'defense_clearances': 0.15,
            'misc_air_dual_won': 0.15,  # ↓ from 0.20 - raw count less important
            'misc_pct_air_dual_won': 0.10,  # NEW - quality of aerial duels
            'passing_pass_prog': 0.20,  # ↓ from 0.25 - still important but rebalanced
            'possession_ttl_carries_prog_dist': 0.15,  # ↓ from 0.20 - reduce over-influence
            'defense_int': 0.15,
            'defense_def_error': -0.15,
        },
        'stats': ['defense_clearances', 'misc_air_dual_won', 'misc_pct_air_dual_won', 'passing_pass_prog',
                  'possession_ttl_carries_prog_dist', 'defense_int', 'defense_def_error']
    },
    'CB': {
        'weights': {
            'defense_clearances': 0.25,  # ↓ from 0.30 - rebalanced
            'misc_air_dual_won': 0.20,  # ↓ from 0.30 - raw count less important
            'misc_pct_air_dual_won': 0.15,  # NEW - quality of aerial duels
            'defense_int': 0.20,  # ↓ from 0.25 - rebalanced
            'defense_def_error': -0.15,
            'blocks_sh_blocked': 0.05,  # NEW - shot blocking
        },
        'stats': ['defense_clearances', 'misc_air_dual_won', 'misc_pct_air_dual_won',
                  'defense_int', 'defense_def_error', 'blocks_sh_blocked']
    },
    'RCB': {
        'weights': {
            'defense_clearances': 0.15,  # ↓ from 0.20 - rebalanced
            'misc_air_dual_won': 0.15,  # ↓ from 0.25 - raw count less important
            'misc_pct_air_dual_won': 0.10,  # NEW - quality of aerial duels
            'defense_tkl_def_third': 0.20,  # ↓ from 0.25 - rebalanced
            'defense_def_error': -0.15,
            'passing_pct_pass_cmp': 0.15,
            'defense_blocks': 0.10,  # NEW - blocking
        },
        'stats': ['defense_clearances', 'misc_air_dual_won', 'misc_pct_air_dual_won', 'defense_tkl_def_third',
                  'defense_def_error', 'passing_pct_pass_cmp', 'defense_blocks']
    },
    # Wing-Backs (attack/defense balance)
    'LWB': {
        'weights': {
            'possession_ttl_carries_prog_dist': 0.20,  # ↓ from 0.30 - reduce over-influence
            'defense_tkl': 0.15,  # ↓ from 0.20 - rebalanced
            'passing_cross_opp_box': 0.20,  # ↓ from 0.25 - rebalanced
            'gca_ttl_sca': 0.15,
            'defense_def_error': -0.10,
            'touch_fthird': 0.15,  # NEW - final third presence
            'carries_fthird': 0.15,  # NEW - carries into final third
        },
        'stats': ['possession_ttl_carries_prog_dist', 'defense_tkl',
                  'passing_cross_opp_box', 'gca_ttl_sca', 'defense_def_error',
                  'touch_fthird', 'carries_fthird']
    },
    'RWB': {
        'weights': {
            'defense_tkl_drb': 0.20,  # ↓ from 0.25 - rebalanced
            'passing_cross_opp_box': 0.20,  # ↓ from 0.30 - rebalanced
            'gca_ttl_sca': 0.15,  # ↓ from 0.20 - rebalanced
            'defense_tkl_att_third': 0.15,  # High pressing
            'defense_def_error': -0.10,
            'touch_fthird': 0.15,  # NEW - final third presence
            'carries_fthird': 0.15,  # NEW - carries into final third
        },
        'stats': ['defense_tkl_drb', 'passing_cross_opp_box', 'gca_ttl_sca',
                  'defense_tkl_att_third', 'defense_def_error', 'touch_fthird', 'carries_fthird']
    },
    # Central Midfielders
    'LCM': {
        'weights': {
            'defense_tkl_plus_int': 0.25,  # ↓ from 0.30 - rebalanced
            'misc_ball_recov': 0.20,  # ↓ from 0.25 - rebalanced
            'passing_pct_pass_cmp': 0.15,  # ↓ from 0.20 - rebalanced
            'defense_def_error': -0.15,
            'possession_pass_recvd': 0.10,
            'touch_mid_third': 0.15,  # NEW - midfield presence
            'defense_tkl_mid_third': 0.10,  # NEW - midfield tackling
        },
        'stats': ['defense_tkl_plus_int', 'misc_ball_recov', 'passing_pct_pass_cmp',
                  'defense_def_error', 'possession_pass_recvd', 'touch_mid_third',
                  'defense_tkl_mid_third']
    },
    'RCM': {
        'weights': {
            'passing_pass_prog': 0.25,  # ↓ from 0.35 - rebalanced
            'gca_pass_live_sca': 0.20,  # ↓ from 0.25 - rebalanced
            'defense_tkl_mid_third': 0.15,
            'stats_gls_and_ast': 0.15,
            'defense_def_error': -0.10,
            'passing_key_passes': 0.15,  # NEW - creative influence
            'touch_fthird': 0.10,  # NEW - final third presence
        },
        'stats': ['passing_pass_prog', 'gca_pass_live_sca', 'defense_tkl_mid_third',
                  'stats_gls_and_ast', 'defense_def_error', 'passing_key_passes',
                  'touch_fthird']
    },
    # Attacking Midfielder
    'CAM': {
        'weights': {
            'gca_ttl_sca': 0.25,  # ↓ from 0.30 - rebalanced
            'passing_key_passes': 0.20,  # ↓ from 0.25 - rebalanced
            'passing_xa': 0.15,  # ↓ from 0.20 - rebalanced
            'possession_take_on_suc': 0.10,  # ↓ from 0.15 - rebalanced
            'shooting_gls_xg_diff': 0.10,
            'touch_opp_box': 0.15,  # NEW - box presence
            'gca_pass_live_gca': 0.05,  # NEW - direct goal creation
        },
        'stats': ['gca_ttl_sca', 'passing_key_passes', 'passing_xa',
                  'possession_take_on_suc', 'shooting_gls_xg_diff',
                  'touch_opp_box', 'gca_pass_live_gca']
    },
    # Forwards
    'LW': {
        'weights': {
            'possession_take_on_suc': 0.25,  # ↓ from 0.30 - rebalanced
            'gca_take_on_sca': 0.20,  # ↓ from 0.25 - rebalanced
            'shooting_gls_xg_diff': 0.15,  # ↓ from 0.20 - rebalanced
            'defense_tkl_att_third': 0.10,  # ↓ from 0.15 - rebalanced
            'defense_def_error': -0.10,
            'touch_opp_box': 0.15,  # NEW - box presence
            'stats_gls_and_ast': 0.15,  # NEW - direct goal contributions
        },
        'stats': ['possession_take_on_suc', 'gca_take_on_sca', 'shooting_gls_xg_diff',
                  'defense_tkl_att_third', 'defense_def_error', 'touch_opp_box',
                  'stats_gls_and_ast']
    },
    'RW': {
        'weights': {
            'passing_cross_opp_box': 0.20,  # ↓ from 0.30 - rebalanced
            'gca_pass_live_sca': 0.20,  # ↓ from 0.25 - rebalanced
            'shooting_gls_xg_diff': 0.15,  # ↓ from 0.20 - rebalanced
            'possession_take_on_suc': 0.15,
            'defense_def_error': -0.10,
            'touch_opp_box': 0.15,  # NEW - box presence
            'stats_gls_and_ast': 0.15,  # NEW - direct goal contributions
        },
        'stats': ['passing_cross_opp_box', 'gca_pass_live_sca', 'shooting_gls_xg_diff',
                  'possession_take_on_suc', 'defense_def_error', 'touch_opp_box',
                  'stats_gls_and_ast']
    },
    'ST': {
        'weights': {
            'stats_gls': 0.25,  # ↓ from 0.35 - rebalanced
            'shooting_gls_xg_diff': 0.20,  # ↓ from 0.30 - rebalanced
            'misc_air_dual_won': 0.15,  # ↓ from 0.20 - rebalanced
            'gca_ttl_sca': 0.10,
            'defense_tkl_att_third': 0.10,  # ↑ from 0.05 - pressing importance
            'touch_opp_box': 0.15,  # NEW - box presence
            'shooting_sh': 0.05,  # NEW - shot volume
        },
        'stats': ['stats_gls', 'shooting_gls_xg_diff', 'misc_air_dual_won',
                  'gca_ttl_sca', 'defense_tkl_att_third', 'touch_opp_box',
                  'shooting_sh']
    }
}


def calculate_position_scores(player_data):
    """
    Calculate position scores for a player based on their stats and position weights.

    Args:
        player_data (dict): Dictionary containing player statistics

    Returns:
        dict: Dictionary with position scores for all positions (except GK)
    """
    try:
        minutes = float(player_data.get('stats_min', 1))  # Avoid division by zero

        # Calculate position scores
        position_scores = {}

        # Skip goalkeeper position
        for position, config in POSITION_WEIGHTS.items():
            if position == 'GK':
                continue

            score = 0
            missing_stats = []

            for stat, weight in config['weights'].items():
                if stat not in player_data:
                    missing_stats.append(stat)
                    continue

                if player_data[stat] is None:
                    player_data[stat] = 0

                # Convert to per-90 (ignore percentages, ratios, and per90 stats)
                # Define the stats that should NOT be converted to per90
                non_per90_stats = [
                    'passing_pct_pass_cmp',
                    'shooting_gls_xg_diff',
                    'misc_pct_air_dual_won',
                    'possession_take_on_suc'
                ]
                non_per90_suffixes = ['_pct', '_per90', '_diff', '_suc']

                is_non_per90 = (stat in non_per90_stats or
                                any(stat.endswith(suffix) for suffix in non_per90_suffixes))

                if not is_non_per90:
                    # Apply per-90 conversion
                    value = (player_data.get(stat, 0) / (minutes / 90)) if minutes > 0 else 0
                else:
                    # Use raw value for percentage and ratio stats
                    value = player_data.get(stat, 0)

                # Apply log transformation to high-volume metrics to reduce outsized influence
                high_volume_stats = ['possession_ttl_carries_prog_dist', 'passing_pass_prog']
                if stat in high_volume_stats and value > 0:
                    import math
                    # Using log(1+x) to handle zero values
                    value = math.log(1 + value) * 5  # Scale factor to make it comparable to other metrics

                score += value * weight

            if missing_stats:
                print(
                    f"⚠️ {position}: Missing {missing_stats} for {player_data.get('player_name', 'Unknown')} (partial score)")

            position_scores[position] = round(score, 2)

        return position_scores

    except Exception as e:
        print(e)
        return dict()


def main():
    """Display all stats used in the position weights configuration."""
    # Filter out GK position
    position_weights_no_gk = {k: v for k, v in POSITION_WEIGHTS.items() if k != 'GK'}

    all_stats = []
    for k, v in position_weights_no_gk.items():
        all_stats.extend(v['stats'])

    # Also include all specific weight keys
    for k, v in position_weights_no_gk.items():
        all_stats.extend(list(v['weights'].keys()))

    # Remove duplicates
    all_stats = list(set(all_stats))
    pprint(all_stats)
    print(f"Total unique stats used: {len(all_stats)}")


if __name__ == '__main__':
    main()
