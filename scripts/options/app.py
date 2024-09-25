import requests
import time
import hmac
import hashlib
import os


base_url = 'https://www.deribit.com/api/v2/'

def get_active_instruments(currency='BTC', kind='option'):
    url = f'{base_url}public/get_instruments'
    params = {
        'currency': currency,
        'kind': kind,
        'expired': 'false'
    }
    response = requests.get(url, params=params)
    return response.json()

# Obter a lista de opções BTC ativas
instruments = get_active_instruments()
print(instruments)


"""
api_key = os.getenv('DERIBIT_CLIENT_IDENTITY')
api_secrect = os.getenv('DERIBIT_CLIENT_SECRET')
base_url = 'https://www.deribit.com/api/v2/'

def get_ticker(symbol):
    url = f'{base_url}public/ticker?instrument_name={symbol}'
    response = requests.get(url)
    return response.json()

# Exemplo de chamada para obter dados de uma opção BTC
ticker_data = get_ticker('BTC-24SEP21-45000-P')
print(ticker_data)
"""