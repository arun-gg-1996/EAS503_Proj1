import os
import time

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from pprint import pprint

dir_path = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(dir_path, '.env')

load_dotenv(env_path)
api_key = os.getenv("API_KEY")


def team_stats(league_id):
    url = "https://fbrapi.com/team-season-stats"
    params = {
        "league_id": league_id
    }
    headers = {"X-API-Key": api_key}

    response = requests.get(url, params=params, headers=headers)
    teams = response.json().get('data', [])
    teams = [team.get('meta_data', {}) for team in teams]
    return teams


def main():
    teams = list()
    leagues = [{'country_code': 'ENG', 'league_id': 9, 'league_name': 'Premier League'},
               {'country_code': 'GER', 'league_id': 20, 'league_name': 'Fußball-Bundesliga'},
               {'country_code': 'FRA', 'league_id': 13, 'league_name': 'Ligue 1'},
               {'country_code': 'ITA', 'league_id': 11, 'league_name': 'Serie A'},
               {'country_code': 'ESP', 'league_id': 12, 'league_name': 'La Liga'}]
    for league in tqdm(leagues):
        league_data = team_stats(league_id=league['league_id'])
        time.sleep(4)
        for team in league_data:
            team['league_id'] = league['league_id']
            team['league_name'] = league['league_name']

        print(len(league_data), league['country_code'])
        teams.extend(league_data)

    print(len(teams))

if __name__ == '__main__':
    main()
