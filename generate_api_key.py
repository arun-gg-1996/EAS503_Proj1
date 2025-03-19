import os

import requests
from dotenv import load_dotenv, set_key

from consts import API_KEY

dir_path = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(dir_path, '.env')

load_dotenv(env_path)
api_key = os.getenv("API_KEY")

if not api_key:
    response = requests.post('https://fbrapi.com/generate_api_key')
    api_key = response.json()['api_key']
    set_key(env_path, API_KEY, api_key)
