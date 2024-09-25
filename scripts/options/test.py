import requests
import datetime

# Endpoint da API do Deribit para listar as opções de BTC
url = "https://test.deribit.com/api/v2/public/get_instruments?currency=BTC&kind=option&expired=false"

# Fazendo a requisição à API
response = requests.get(url)

# Verificando se a requisição foi bem-sucedida
if response.status_code == 200:
    data = response.json()

    # Filtrando as opções pela diferença de 1 dia entre a Expiry Date e a Creation Date
    options = data['result']
    print(f"Total de opções retornadas: {len(options)}")
    
    for option in options:
        # Convertemos os timestamps de criação e expiração para objetos datetime
        creation_date = datetime.datetime.fromtimestamp(option['creation_timestamp'] / 1000)
        expiry_date = datetime.datetime.fromtimestamp(option['expiration_timestamp'] / 1000)
        
        # Calculando a diferença entre as datas
        date_diff = (expiry_date - creation_date).days
        
        # Filtrando as opções com diferença de 1 dia
        if date_diff == 2:
            print(f"Option: {option['instrument_name']}")
            print(f"Strike: {option['strike']}")
            print(f"Expiry Date: {expiry_date.strftime('%Y-%m-%d')}")
            print(f"Creation Date: {creation_date.strftime('%Y-%m-%d')}")
            print(f"Option Type: {option['option_type']}")
            print(f"Settlement: {option['settlement_period']}")
            print(f"Days between Creation and Expiry: {date_diff}")
            print("-----------------------------")
else:
    print(f"Erro ao obter os dados: {response.status_code}")
