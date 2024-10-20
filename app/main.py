import backtrader as bt
import pandas as pd
import numpy as np

from yfinance_data import YFinance
from sklearn.ensemble import RandomForestRegressor
from indicadores import Indicadores
from MLModels import using_RandomForestRegressor
from backtest import Test
from option_price import black_scholes_call, black_scholes_put, garch_estimate_volatility

def main():
    START_DATE = '2018-01-01'
    TRADE_START_DATE = '2022-01-01'  
    END_DATE = '2023-12-30'
    INITIAL_CASH = 10000
    r = 0.01 / 365
    T = 7 / 365

    data = YFinance.get_data_from_date('BTC-USD', START_DATE, END_DATE)

    data = precify_options(data, r, T)

    
    data = ml_model(data)

    # Definir uma classe personalizada para incluir a coluna 'signal_ml'
    class PandasData(bt.feeds.PandasData):
        lines = ('signal_ml',)  # Adiciona 'signal_ml' como um novo campo
        params = (('signal_ml', -1),)  # Define o índice de 'signal_ml' como -1 (auto)

    # Definir a classe da estratégia
    class MLStrategy(bt.Strategy):
        params = (
            ('trade_start_date', TRADE_START_DATE),
        )

        def __init__(self):
            self.dataclose = self.datas[0].close
            self.signal = self.datas[0].signal_ml
            self.start_trading = False  # Controle para iniciar operações a partir da data

        def next(self):
            # Verificar se a data atual é posterior ou igual à data de início das operações
            current_date = self.datas[0].datetime.date(0)
            if current_date >= pd.to_datetime(self.params.trade_start_date).date():
                self.start_trading = True

            # Somente operar se a data atual for posterior à data de início
            if not self.start_trading:
                return

            # Se sinal for de compra (1) e não houver posição, compre tudo
            if self.signal[0] == 1 and not self.position:
                self.buy(size=self.broker.get_cash() / self.dataclose[0])
            # Se sinal for de venda (-1) e houver posição, venda tudo
            elif self.signal[0] == -1 and self.position:
                self.sell(size=self.position.size)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(MLStrategy, trade_start_date=TRADE_START_DATE)

    # Adicionar os dados do Pandas ao Backtrader com a nova classe PandasData
    datafeed = PandasData(dataname=data)
    cerebro.adddata(datafeed)

    cerebro.broker.set_cash(INITIAL_CASH)

    cerebro.run()

    data_copy = data.copy()
    accuracy, total_signals, correct_signals = Test().check_signal_accuracy(data_copy)

    print(f'Valor final do portfólio: {cerebro.broker.getvalue()}')

    cerebro.plot()
    
    data.to_csv('data.csv', index=False)


def precify_options(data, r, T):
    # Calcular a volatilidade anualizada
    data['volatility'] = data['close'].rolling(window=30).apply(lambda x: garch_estimate_volatility(x) * np.sqrt(365), raw=False)

    data.dropna(subset=['volatility'], inplace=True)

    data['call_price'] = data.apply(lambda row: black_scholes_call(
        S=row['close'],
        K=row['close'],
        T=T,
        r=r,
        sigma=row['volatility']
    ), axis=1)

    data['put_price'] = data.apply(lambda row: black_scholes_put(
        S=row['close'],
        K=row['close'],
        T=T,
        r=r,
        sigma=row['volatility']
    ), axis=1)

    return data



def ml_model(data):
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # Preparação das features e alvo para o modelo de Machine Learning
    data['target'] = data['close'].shift(-1)
    features = ['open', 'high', 'low', 'close', 'volume']

    # Calcular indicadores técnicos e adicionar às features
    indicadores = Indicadores()
    data['rsi_14'] = indicadores.get_rsi(data['close'])
    features.append('rsi_14')
    data['stoch_rsi_14'] = indicadores.get_stochastic_rsi(data['close'])
    features.append('stoch_rsi_14')
    data['macd'] = indicadores.get_macd(data['close'])
    features.append('macd')
    data['ema_14'] = indicadores.get_media_movel_exponecial(data['close'])
    features.append('ema_14')

    # Remover valores nulos
    data.dropna(inplace=True)

    # Preparar os dados para o modelo de Machine Learning
    X = data[features]
    y = data['target']

    # Treinar o modelo usando GridSearchCV
    model, data['mean_predictions'], data['std_predictions'] = using_RandomForestRegressor.GridSearchCV_RandomForestRegressor(X, y)
    data['confidence'] = 1 / (1 + data['std_predictions'])

    # Fazer previsões
    data['predicted_close'] = model.predict(X)

    # Gerar sinais de compra/venda com base nas previsões
    data['signal_ml'] = np.where(data['predicted_close'] > data['close'], 1, -1)

    return data

    
if __name__ == '__main__':
    main()