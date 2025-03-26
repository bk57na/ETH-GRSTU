import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def sp500_download(start_date, end_date):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)
    sp500_df = sp500_table[0]
    tickers = sp500_df['Symbol'].tolist()

    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

    output_dir = Path('sp500')
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f'sp500_{end_date}.csv'
    data.to_csv(filename)
    print(f'Data saved to {filename}')

def download_index_group(tickers, group_name, start_date, end_date):
    combined_data = {}

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            combined_data[ticker] = data

    full_df = pd.concat(combined_data, axis=1)

    output_dir = Path('indices')
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f'{group_name}_{end_date}.csv'
    full_df.to_csv(filename)
    print(f'Data saved to {filename}')


def main():
    start_date = '1970-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')

    index_groups = {
        "Core Market Indicators": [
            '^GSPC', '^FTSE', '^N225', '^HSI',
            '^GDAXI', '^STOXX50E', '^BVSP',
            'GC=F', 'CL=F', '^TNX', '^VIX'
        ],
        "Geographically Segmented": [
            'EUSA', 'EFA', 'EEM', 'FM',
            'BNDW', 'GSG'
        ]
    }

    for group_name, tickers in index_groups.items():
        download_index_group(tickers, group_name, start_date, end_date)

if __name__ == '__main__':
    main()