import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

if not mt5.initialize():
    print("Falha ao inicializar MetaTrader5")
    mt5.shutdown()

symbol = "BTCUSD"

if not mt5.symbol_select(symbol, True):
    print(f"Falha ao selecionar símbolo {symbol}")
    mt5.shutdown()

# Obter dados históricos de preço (exemplo: últimas 24 horas)
#rates_h = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 48)
rates_d = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 1500)

mt5.shutdown()

#rates_h_frame = pd.DataFrame(rates_h)
rates_d_frame = pd.DataFrame(rates_d)

# Converter a coluna de tempo de UNIX timestamp para datetime legível
#rates_h_frame['time'] = pd.to_datetime(rates_h_frame['time'], unit='s')
rates_d_frame['time'] = pd.to_datetime(rates_d_frame['time'], unit='s')


#print(rates_h_frame[['time', 'open', 'close', 'high', 'low']])
print(rates_d_frame[['time', 'open', 'close', 'high', 'low']])

#rates_h_frame.to_csv('rates_h_frame.csv', index=False)
rates_d_frame.to_csv('rates_d_frame.csv', index=False)

from IndicadoresMercado import Indicadores
from MLModels import using_RandomForestRegressor
from Test import Test
import numpy as np

rates_d_frame['target_close'] = rates_d_frame['close'].shift(-1)
rates_d_frame['target_open_d2'] = rates_d_frame['open'].shift(-2)

#rates_d_frame[['time', 'open', 'close', 'high', 'low', 'tick_volume','target_close']].to_csv('rates_d_frame_target.csv', index=False)
features = ['open', 'close', 'high', 'low', 'tick_volume']


rates_d_frame[f'SMA_9'] = Indicadores.Media_movel(rates_d_frame['close'], window=9)
features.append('SMA_9')

rates_d_frame['RSI'] = Indicadores.compute_RSI(rates_d_frame['close'], window=14)
features.append('RSI')

macd_histogram = Indicadores().compute_MACD(rates_d_frame['close'], short_window=12, long_window=26, signal_window=9)
rates_d_frame['macd_histogram'] = macd_histogram
features.append('macd_histogram')

rates_d_frame.dropna(inplace=True)

X = rates_d_frame[features]
y = rates_d_frame['target_close']


model, rates_d_frame['mean_predictions'], rates_d_frame['std_predictions']= using_RandomForestRegressor.GridSearchCV_RandomForestRegressor(X, y)
rates_d_frame['confidence '] = 1 / (1 + rates_d_frame['std_predictions'])

rates_d_frame['predicted_close']= model.predict(X)

rates_d_frame['signal_ml'] = np.where(rates_d_frame['predicted_close'] > rates_d_frame['close'], 1, -1)

rates_d_frame.to_csv('signal_ml.csv', index=False)

accuracy = Test.check_signal_accuracy(rates_d_frame)
print(f'Acurácia do sinal: {accuracy:.2f}')
print('\n')


import backtrader as bt
class PandasData(bt.feeds.PandasData):
    lines = ('signal_ml',)
    params = (('signal_ml', -3),)

data_feed = PandasData(dataname=rates_d_frame, datetime='time', open='open', high='high', low='low', close='close', volume='tick_volume')

class MLStrategy_two(bt.Strategy):
    params = (
        ('start_date', datetime(2023, 9, 21)),  # Data para começar a estratégia
        ('risk_per_trade', 1.0),
    )

    def __init__(self):
        self.start_trading = False
        self.buy_price = None
        self.sell_price = None
        self.current_value = self.broker.get_cash()

    def next(self):
        current_date = self.data.datetime.date(0)
        close_price = self.data.close[0]
        
        # Aplicar slippage
        self.buy_price = close_price * 1.001
        self.sell_price = close_price * 0.999

        # Verificar se a data atual é igual ou maior que a data de início
        if current_date >= self.params.start_date.date():
            self.start_trading = True

        if self.start_trading:
            if self.data.signal_ml[0] == 1 and self.broker.get_cash() > 0:
                amount_to_risk = self.broker.get_cash() * self.params.risk_per_trade
                size = amount_to_risk / self.buy_price
                self.buy(size=size)
                
            elif self.data.signal_ml[0] == -1 and self.position:
                self.sell(size=self.position.size)
        
        # Atualiza o valor atual do portfólio
        self.current_value = self.broker.get_value()

class MLStrategy_one(bt.Strategy):
    params = (
        ('start_date', datetime(2023, 8, 27)),  # Data para começar a estratégia
    )

    def __init__(self):
        self.start_trading = False  # Controle para iniciar a estratégia

    def next(self):
        current_date = self.data.datetime.date(0)

        if current_date >= self.params.start_date.date():
            self.start_trading = True

        if self.start_trading:
            signal_ml = self.data.signal_ml[0]  # Obtendo o sinal da coluna 'action'
            if signal_ml == 1 and not self.position:
                self.buy()  # Compra se o sinal for 'Buy'
            elif signal_ml == -1 and self.position:
                self.sell()  # Vende se o sinal for 'Sell'


class RiskSizer(bt.Sizer):
    params = (
        ('risk_per_trade', 1.0),  # Percentual do capital a arriscar em cada trade
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:  # Para compras
            amount_to_risk = cash * self.params.risk_per_trade
            size = amount_to_risk / data.close[0]  # Divide pelo preço do ativo para determinar a quantidade
            return size
        else:  # Para vendas, vende tudo que tem
            return self.broker.getposition(data).size


# Configurando o ambiente de backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(MLStrategy_two)

# Adicionando os dados ao cerebro
cerebro.adddata(data_feed)

# Definindo o capital inicial
cerebro.broker.set_cash(10000)

# Definindo a comissão (não usada diretamente aqui, mas pode ser adicionada)
# Configurando a execução no mesmo candle e slippage


# Executando o backtest
print(f'MLStrategy_two: Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.run()
print(f'MLStrategy_two: Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
print(f'Rentabilidade = {round(((cerebro.broker.getvalue() - 10000) / 10000) * 100, 2)} %',  )

# Exibindo o gráfico
cerebro.plot()
