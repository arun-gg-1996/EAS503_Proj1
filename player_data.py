import os

import requests
from dotenv import load_dotenv
from pprint import pprint

dir_path = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(dir_path, '.env')

load_dotenv(env_path)
api_key = os.getenv("API_KEY")


def get_players(team_id, league_id, season_id):
    url = "https://fbrapi.com/player-season-stats"
    params = {
        "team_id": team_id,
        "league_id": league_id,
        "season_id": season_id,

    }

    headers = {"X-API-Key": api_key}
    response = requests.get(url, params=params, headers=headers)

    players = [each for each in response.json().get("players", []) if each.get('meta_data', {}).get('player_id', None)]
    keepers = [each for each in response.json().get("keepers", []) if each.get('meta_data', {}).get('player_id', None)]

    # TODO - undo
    # return players + keepers
    return players


def main():
    players = get_players(team_id='18bb7c10', league_id=9, season_id='2023-2024')
    pprint(players)


if __name__ == '__main__':
    main()
