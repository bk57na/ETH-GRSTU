import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def sp500_download():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)
    sp500_df = sp500_table[0]
    tickers = sp500_df['Symbol'].tolist()

    start_date = '1970-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

    Path('data').mkdir(exist_ok=True)
    data.to_csv(f'data/sp500_{end_date}.csv')
    print('Data saved to data/sp500.csv')

if __name__ == '__main__':
    sp500_download()