import pandas as pd
import requests


def load_wmt_eod():
    df = pd.read_csv('../input/wmt_eod.csv')
    df.index = pd.to_datetime(df['Date'])
    df.drop(['Date'], axis=1, inplace=True)
    return df


def load_wmt_fund():
    df = pd.read_csv('../input/wmt_fund.csv')
    df = df.dropna(axis=1, how='all')
    df = remove_zero_cols(df)
    df['reportperiod'] = pd.to_datetime(df['reportperiod'])
    return df


def load_wmt_pre():
    df = pd.read_csv('../input/wmt_pre.csv')
    df = df[df['per_type'] == 'Q']
    df = df[df['est_type_desc'] == 'EPS']
    df.index = pd.to_datetime(df['per_end_date'])
    df['announce_date'] = pd.to_datetime(df['announce_date'])
    df = df[['per_end_date', 'announce_date', 'pa_low',
             'pa_high', 'pa_mean', 'pa_amt_diff_surp', 'pa_pct_diff_surp']]
    return df


def get_csv_from_quandl(url, file_path):
    r = requests.get(url)
    with open(file_path, 'wb+') as f:
        f.write(r.content)


def get_wmt_fundamentals():
    file_path = '../input/wmt_fund.csv'
    url = 'https://www.quandl.com/api/v3/datatables/SHARADAR/SF1.csv?'\
          'ticker=WMT&dimension=ARQ&api_key=GgxeN_cyttYzSsZAqiT9'
    get_csv_from_quandl(url, file_path)
    return pd.read_csv(file_path)


def get_wmt_eod():
    file_path = '../input/wmt_eod.csv'
    url = 'https://www.quandl.com/api/v3/datasets/EOD/WMT.csv?'\
          'api_key=GgxeN_cyttYzSsZAqiT9'
    get_csv_from_quandl(url, file_path)
    return pd.read_csv(file_path)


def get_wmt_pre():
    file_path = '../input/wmt_pre.csv'
    url = 'https://www.quandl.com/api/v3/datatables/ZACKS/IRH.csv?'\
          'ticker=WMT&api_key=GgxeN_cyttYzSsZAqiT9'
    get_csv_from_quandl(url, file_path)
    return pd.read_csv(file_path)


def remove_zero_cols(df):
    return df.loc[:, (df != 0).any(axis=0)]
