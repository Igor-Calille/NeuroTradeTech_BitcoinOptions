import numpy as np
import pandas as pd
from datetime import datetime


class Test:
    def __init__ (self):
        pass

    #Função para calcular a acurácia do sinal de trading
    def check_signal_accuracy(data):
        # Calcular a mudança de preço do dia seguinte
        data['price_change'] = data['close'].shift(-1) - data['close']

        # Definir condições para um sinal correto de compra ou venda
        conditions = [
            (data['signal_ml'] > 0) & (data['price_change'] > 0),  # Compra seguida de aumento de preço
            (data['signal_ml'] < 0) & (data['price_change'] < 0),  # Venda seguida de queda de preço
        ]
        choices = [1, 1]  # Ambos são sinais corretos, independente de serem compra ou venda
        data['correct_signal'] = np.select(conditions, choices, default=0)

        # Calcular a acurácia do sinal
        correct_signals = data['correct_signal'].sum()
        total_signals = np.count_nonzero(data['signal_ml'])  # Conta todos os sinais emitidos, ignorando zeros
        accuracy = correct_signals / total_signals if total_signals > 0 else 0  # Evita divisão por zero
        return accuracy
    
    def backtest_signals(data, initial_capital=10000, risk_per_trade=1.0):
        cash = initial_capital
        position = 0
        portfolio_value = []

        for index, row in data.iterrows():
            buy_price = row['close'] * 1.01  # Inclui slippage
            sell_price = row['close'] * 0.99  # Inclui slippage
            amount_to_risk = cash * risk_per_trade

            if row['action'] == 'Buy' and cash > 0:
                shares_to_buy = amount_to_risk / buy_price
                position += shares_to_buy
                cash -= shares_to_buy * buy_price

            elif row['action'] == 'Sell' and position > 0:
                cash += position * sell_price
                position = 0

            current_value = cash + position * row['close']
            portfolio_value.append(current_value)

        data['Portfolio Value'] = portfolio_value
        return data['Portfolio Value'].iloc[-1], data
    
    def backtest_signals_date(data, date_str, initial_capital=10000, risk_per_trade=1.0):
        cash = initial_capital
        position = 0
        portfolio_value = []
        date = pd.to_datetime(date_str)
        count_trades = 0
        current_value = cash  # Valor inicial do portfólio

        for index, row in data.iterrows():
            # Manter o valor inicial até a data especificada
            if pd.to_datetime(row['date']) < date:
                portfolio_value.append(current_value)
                continue

            # Realizar trades quando a data é maior ou igual à data especificada
            count_trades = count_trades + 1
            buy_price = row['close'] * 1.01  # Inclui slippage
            sell_price = row['close'] * 0.99  # Inclui slippage
            amount_to_risk = cash * risk_per_trade

            if row['action'] == 'Buy' and cash > 0:
                shares_to_buy = amount_to_risk / buy_price
                position += shares_to_buy
                cash -= shares_to_buy * buy_price

            elif row['action'] == 'Sell' and position > 0:
                cash += position * sell_price
                position = 0

            current_value = cash + position * row['close']
            portfolio_value.append(current_value)

        # Garantir que a lista portfolio_value tenha o mesmo comprimento que o DataFrame data
        data['Portfolio Value'] = portfolio_value
        return data['Portfolio Value'].iloc[-1], data, count_trades
    
    def backtest_signals_date_rpt(data, date, initial_capital, risk_per_trade=1.0, slippage=0.0):
        cash = initial_capital
        current_value = initial_capital
        position = 0
        date = pd.to_datetime(date)
        count_signals = 0
        up_slippage = 1 + slippage
        down_slippage = 1 - slippage
        
        # Iniciando a coluna do valor da carteira
        data['current_value'] = initial_capital

        for index, row in data.iterrows():
            if pd.to_datetime(row['date']) >= date:
                
                buy_price = row['close'] * up_slippage
                sell_price = row['close'] * down_slippage
                amount_to_risk = cash * risk_per_trade

                if row['signal'] == 1 and cash > 0:
                    #shares_to_buy = (amount_to_risk / buy_price) * abs(row['bollinger_low_proportion_20'])
                    shares_to_buy = (amount_to_risk / buy_price) 
                    position += shares_to_buy
                    cash -= shares_to_buy * buy_price
                    count_signals += 1

                elif row['signal'] == -1 and position > 0:
                    #shares_to_sell = position * abs(row['bollinger_high_proportion_20'])
                    shares_to_sell = position
                    cash += shares_to_sell * sell_price
                    position -= shares_to_sell
                    count_signals += 1

                current_value = cash + position * row['close']
                data.at[index, 'current_value'] = current_value
            else:
                if index > 0:
                    data.at[index, 'current_value'] = data.at[index-1, 'current_value']
                else:
                    data.at[index, 'current_value'] = initial_capital
        
        # Preenchendo os valores restantes após a última trade
        for index in range(len(data)-1):
            if data.at[index+1, 'current_value'] == initial_capital:
                data.at[index+1, 'current_value'] = data.at[index, 'current_value']
        
        return data, count_signals

    def backtest_signals_SL_TP(data, initial_capital=10000, risk_per_trade=0.02, stop_loss_percent=0.05, take_profit_percent=0.10):
        cash = initial_capital
        position = 0
        portfolio_value = []
        entry_price = 0

        for index, row in data.iterrows():
            buy_price = row['close'] * 1.01
            sell_price = row['close'] * 0.99
            stop_loss_price = entry_price * (1 - stop_loss_percent)
            take_profit_price = entry_price * (1 + take_profit_percent)

            if row['action'] == 'Buy' and cash > 0:
                position = (cash * risk_per_trade) / buy_price
                cash -= position * buy_price
                entry_price = buy_price
            elif row['action'] == 'Sell' and position > 0:
                if sell_price <= stop_loss_price or sell_price >= take_profit_price:
                    cash += position * sell_price
                    position = 0

            current_value = cash + position * row['close']
            portfolio_value.append(current_value)

        data['Portfolio Value'] = portfolio_value
        return data['Portfolio Value'].iloc[-1], data




