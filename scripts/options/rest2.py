import requests
from datetime import datetime, timedelta

# Configurações da API
base_url = "https://www.deribit.com/api/v2"

##
# Autenticação
def get_access_token():
    auth_url = f"{base_url}/public/auth"
    data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    response = requests.post(auth_url, json=data)
    return response.json().get("result", {}).get("access_token")

# Buscar contratos futuros de Bitcoin
def get_future_contracts(access_token):
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    params = {
        'currency': 'BTC',
        'kind': 'future',
        'expired': 'false'
    }
    response = requests.get(f"{base_url}/public/get_instruments", params=params, headers=headers)
    return response.json().get("result", [])

# Filtrar contratos com vencimento 1 dia após a data de criação
def filter_one_day_expiration(contracts):
    one_day_ms = 24 * 60 * 60 * 1000  # 24 horas em milissegundos
    valid_contracts = []
    for contract in contracts:
        creation_timestamp = contract.get('creation_timestamp')
        expiration_timestamp = contract.get('expiration_timestamp')
        if creation_timestamp and expiration_timestamp:
            # Verifica se a diferença entre a criação e o vencimento é exatamente 1 dia
            if expiration_timestamp - creation_timestamp == one_day_ms:
                valid_contracts.append(contract)
    return valid_contracts

# Execução
access_token = get_access_token()
future_contracts = get_future_contracts(access_token)



for i in future_contracts:
    print(i)