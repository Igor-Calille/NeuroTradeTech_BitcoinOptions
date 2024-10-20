import yfinance as yf

class YFinance:
    def get_data_from_date(symbol, start_date, end_date=None):

        data = yf.download(symbol, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        data = data.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high', 
                'Low': 'low', 'Close': 'close', 'Volume': 'volume', 
                'Adj Close': 'adj_close'
            })
        
        return data
    
    def get_data_from_date_w(symbol, start_date, end_date=None):

        data = yf.download(symbol, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        data = data.rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high', 
                'Low': 'low', 'Close': 'close', 'Volume': 'volume', 
                'Adj Close': 'adj_close'
            })
        
        data.set_index('date', inplace=True)
        weekly_data = data.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'adj_close': 'last'
        }).dropna().reset_index()

        
        return weekly_data