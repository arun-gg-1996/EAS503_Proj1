import os
import time
from pprint import pprint

import requests
from dotenv import load_dotenv
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(dir_path, '.env')

load_dotenv(env_path)
api_key = os.getenv("API_KEY")


def get_league_id(country_code, league_name):
    try:
        url = "https://fbrapi.com/leagues"
        params = {
            "country_code": country_code
        }
        headers = {"X-API-Key": api_key}

        response = requests.get(url, params=params, headers=headers)

        leagues = response.json().get('data', [])
        leagues = [league for league in leagues if league['league_type'] == 'domestic_leagues'][0]['leagues']

        for league in leagues:
            if league['competition_name'] == league_name:
                return league['league_id']

        return None

    except Exception as e:
        print(e)
        existing_mapping = {
            'ENG': 9,
            'GER': 20,
            'FRA': 13,
            'ITA': 11,
            'ESP': 12
        }
        return existing_mapping[country_code]


def main():
    leagues_needed = [{"country_code": "ENG", "league_name": "Premier League"},
                      {"country_code": "GER", "league_name": "Fußball-Bundesliga"},
                      {"country_code": "FRA", "league_name": "Ligue 1"},
                      {"country_code": "ITA", "league_name": "Serie A"},
                      {"country_code": "ESP", "league_name": "La Liga"}]

    for league in tqdm(leagues_needed):
        league['league_id'] = get_league_id(league["country_code"], league_name=league["league_name"])
        time.sleep(3)

    pprint(leagues_needed)


if __name__ == '__main__':
    main()
