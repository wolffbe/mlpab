"""Feature engineering + ingestion driver (runs locally, WRITES to platform only)."""
import numpy as np
import pandas as pd
import hopsworks

TRAIN = "data/transactions.csv"
SCORE = "data/score_transactions.csv"

CATS = ['cash_advance','travel','online','electronics','restaurant',
        'health','fuel','grocery','clothing','entertainment']

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi/180
    a = (np.sin((lat2-lat1)*p/2)**2
         + np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2)
    return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))

def load():
    tr = pd.read_csv(TRAIN)
    sc = pd.read_csv(SCORE)
    tr['_set'] = 'train'
    sc['_set'] = 'score'
    sc['is_fraud'] = np.nan
    df = pd.concat([tr, sc], ignore_index=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df, tr

def engineer(df, tr):
    g = tr.groupby('cc_num')['amount']
    card_mean = g.mean()
    card_std = g.std().fillna(1.0).replace(0, 1.0)
    home = tr.groupby('cc_num')[['lat','long']].median()

    df = df.sort_values(['cc_num','datetime']).reset_index(drop=True)
    df['amount'] = df['amount'].astype(float)
    df['log_amount'] = np.log1p(df['amount'])
    df['hour'] = df['datetime'].dt.hour
    df['dow'] = df['datetime'].dt.dayofweek

    df['card_mean_amount'] = df['cc_num'].map(card_mean).fillna(df['amount'].mean())
    df['card_std_amount'] = df['cc_num'].map(card_std).fillna(1.0)
    df['amount_zscore'] = (df['amount'] - df['card_mean_amount']) / df['card_std_amount']

    df['home_lat'] = df['cc_num'].map(home['lat']).fillna(df['lat'].median())
    df['home_long'] = df['cc_num'].map(home['long']).fillna(df['long'].median())
    df['dist_from_home'] = haversine(df['lat'], df['long'], df['home_lat'], df['home_long'])

    df['prev_dt'] = df.groupby('cc_num')['datetime'].shift(1)
    df['prev_lat'] = df.groupby('cc_num')['lat'].shift(1)
    df['prev_long'] = df.groupby('cc_num')['long'].shift(1)
    df['time_since_prev_sec'] = (df['datetime'] - df['prev_dt']).dt.total_seconds().fillna(1e6)
    df['dist_from_prev'] = haversine(df['lat'], df['long'],
                                     df['prev_lat'].fillna(df['lat']),
                                     df['prev_long'].fillna(df['long'])).fillna(0.0)
    df['speed'] = df['dist_from_prev'] / (df['time_since_prev_sec']/3600.0 + 1e-3)

    df['txns_1h'] = 0.0
    df['txns_24h'] = 0.0
    for cc, idx in df.groupby('cc_num').groups.items():
        sub = df.loc[idx].sort_values('datetime')
        s = sub.set_index('datetime')['amount']
        df.loc[sub.index, 'txns_1h'] = s.rolling('1h').count().values
        df.loc[sub.index, 'txns_24h'] = s.rolling('24h').count().values

    for c in CATS:
        df['cat_'+c] = (df['category'] == c).astype(int)
    return df

FEATS = (['amount','log_amount','hour','dow','card_mean_amount','card_std_amount',
          'amount_zscore','dist_from_home','dist_from_prev','speed',
          'time_since_prev_sec','txns_1h','txns_24h'] + ['cat_'+c for c in CATS])

def main():
    df, tr = load()
    df = engineer(df, tr)
    keep_base = ['transaction_id','cc_num','datetime']
    train_fe = df[df._set=='train'][keep_base + FEATS + ['is_fraud']].copy()
    train_fe['is_fraud'] = train_fe['is_fraud'].astype(int)
    score_fe = df[df._set=='score'][keep_base + FEATS].copy()
    print('train_fe', train_fe.shape, 'score_fe', score_fe.shape)
    print('fraud rate train_fe', train_fe.is_fraud.mean())

    proj = hopsworks.login()
    fs = proj.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name='cctxnfe5424', version=1,
        description='Engineered fraud features (labelled history)',
        primary_key=['transaction_id'], event_time='datetime',
        online_enabled=False)
    fg.insert(train_fe, write_options={'wait_for_job': True})
    print('inserted cctxnfe5424')

    sfg = fs.get_or_create_feature_group(
        name='ccscorefe5424', version=1,
        description='Engineered fraud features for scoring set (no label)',
        primary_key=['transaction_id'], event_time='datetime',
        online_enabled=False)
    sfg.insert(score_fe, write_options={'wait_for_job': True})
    print('inserted ccscorefe5424')

    try:
        fs.get_feature_view('cctdfe5424', version=1).delete()
    except Exception:
        pass
    query = fg.select(FEATS + ['is_fraud'])
    fv = fs.create_feature_view(name='cctdfe5424', version=1,
                                query=query, labels=['is_fraud'])
    print('created feature view cctdfe5424', fv.version)

if __name__ == '__main__':
    main()
