class Indicadores:
    def __init__(self) -> None:
        pass

    def get_rsi(self, data_value, window=14):
        diff = data_value.diff(1).dropna()
        gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
        loss = (-diff.where(diff < 0, 0)).rolling(window=window).mean()
        RS = gain / loss

        return 100 - (100 / (1 + RS))
    
    def get_stochastic_rsi(self, data_value, window=14, stochastic_window=14):
        rsi = self.get_rsi(data_value, window)

        min_rsi = rsi.rolling(window=stochastic_window).min()
        max_rsi = rsi.rolling(window=stochastic_window).max()
        stochastica_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)

        return stochastica_rsi
    
    def get_media_movel_exponecial(self, data_value, window=14):
        return data_value.ewm(span=window, adjust=False).mean()
    
    def get_macd(self, data_value, short_window=12, long_window=26, signal_window=9):
        short_ema = data_value.ewm(span=short_window, adjust=False).mean()
        long_ema = data_value.ewm(span=long_window, adjust=False).mean()

        macd_line = short_ema - long_ema

        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

        macd_histogram = macd_line - signal_line

        return macd_histogram
