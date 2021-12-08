
import pathlib
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import pickle

MODEL_FILE = pathlib.Path(__file__).parent.joinpath("baseline_model.cbm")
EXDATA_FILE = pathlib.Path(__file__).parent.joinpath("exdata.csv")
AGG_COLS = ["material_code", "company_code", "country", "region", "manager_code"]

FOLDS = 5

def get_features(df: pd.DataFrame,df_count: pd.DataFrame,df2: pd.DataFrame,exdata: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:

    n_mounths = 12
    start_period = month - pd.offsets.MonthBegin(n_mounths)
    end_period = month - pd.offsets.MonthBegin(1)
   
    df = df.loc[:, :end_period]

    features = pd.DataFrame([], index=df.index)
    features["month"] = month.month
    cols_months = [f"vol_tm{i}" for i in range(n_mounths, 0, -1)]
    features[cols_months] = df.loc[:, start_period:end_period].copy()
   
    features['_mean'] = df.mean(axis=1)
   
    for m in cols_months:
        features[m] = features[m]/features['_mean']
    features[[f"dev_tm"]] = (df2[end_period].copy()/features['vol_tm1']).fillna(0)
    
    for c in exdata.columns:
        features[c] = exdata.loc[end_period,c]
    
    for i in range(n_mounths//3, 0, -1):
        features[f"delta_vol_tm{i}"] = features[f'vol_tm{i}']-features[f'vol_tm{i+1}']
   
    features[f"delta2"] = features[f"delta_vol_tm1"]-features[f"delta_vol_tm2"]
    
    rolling = df.rolling(12, axis=1, min_periods=1)
    features = features.join(rolling.mean().iloc[:, -1].rename("last_year_avg"))
    features = features.join(rolling.max().iloc[:, -1].rename("last_year_max"))
    features = features.join(rolling.std().iloc[:, -1].rename("last_year_std"))
    features = features.join(rolling.var().iloc[:, -1].rename("last_year_var"))
    
    rolling = df.rolling(6, axis=1, min_periods=1)
    features = features.join(rolling.mean().iloc[:, -1].rename("last_half_year_avg"))
    features = features.join(rolling.max().iloc[:, -1].rename("last_half_year_max"))
    features = features.join(rolling.std().iloc[:, -1].rename("last_half_year_std"))
    features = features.join(rolling.var().iloc[:, -1].rename("last_half_year_var"))
    
    rolling = df.rolling(3, axis=1, min_periods=1)
    features = features.join(rolling.mean().iloc[:, -1].rename("last_quarter_avg"))
    features = features.join(rolling.max().iloc[:, -1].rename("last_quarter_max"))
    features = features.join(rolling.std().iloc[:, -1].rename("last_quarter_std"))
    features = features.join(rolling.var().iloc[:, -1].rename("last_quarter_var"))
    
    

    features['e1'] = features[f'vol_tm2']+(features[f'vol_tm1']-features[f'vol_tm2'])*2
    features['e2'] = features[f'vol_tm1']+(features[f'vol_tm1']-features[f'vol_tm2'])
    

    features['count0_last_year'] = df[df==0].iloc[:,-12:].isna().sum(axis=1)
    
    for n in [3,6,9,12]:
        rolling = df.rolling(n, axis=1, min_periods=n-1)
        cur = rolling.sum().iloc[:, -1]
        prev = rolling.sum().iloc[:, -2]
        features[f"delta_sum_months_{n}_cur"] = cur-prev - features[f'vol_tm1']
        features[f"dev_sum_months_{n}"] = df.iloc[:,-n:].mul(np.linspace(0,1,n)-np.linspace(1,0,n),axis=1).sum(axis=1)
        
    features["month"] = month.month
    
    for i in range(2, 0, -1):
        for c in ['last_half_year_avg','last_half_year_max',
                  'last_quarter_avg','last_quarter_max']:
            features[f'{c}_{i}_delta'] = features[c] - features[f"vol_tm{i}"] 
            
    features['_vol_tm1'] =  df[end_period]
    
    return features.reset_index()


def predict(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:

    group_ts = df.groupby(AGG_COLS + ["month"])["volume"].sum().unstack(fill_value=0).apply(lambda x:np.log10(x+1))
    group_ts_count = df.groupby(AGG_COLS + ["month"])["volume"].count().unstack(fill_value=0).apply(lambda x:np.log10(x+1))
    
    group_day = df.copy().groupby(AGG_COLS + ["date"])["volume"].sum().unstack(fill_value=0).stack().rename('vol')
    group_day = group_day.reset_index(level=-1)
    group_day['month'] = group_day.date.apply(lambda x:x.replace(day=1))
    group_day['day'] = group_day.date.apply(lambda x:x.day)
    group_day['max_day'] = group_day.groupby(['month'])['day'].transform('max')
    group_day['max_vol'] = group_day.groupby(['month'])['vol'].transform('max')
    group_day['vol'] /= group_day['max_vol']
    group_day['inv_day'] = group_day['max_day']-group_day['day']+1
    group_day['v'] = group_day['vol']*(group_day['day']-group_day['inv_day'])

    group_ts2 = group_day.reset_index().groupby(AGG_COLS+['month'])['v'].sum().unstack(fill_value=0)
    
    exdata = pd.read_csv(EXDATA_FILE, parse_dates=["month"],index_col='month')
    
    features = get_features(group_ts,group_ts_count,group_ts2,exdata, month)

    model = CatBoostRegressor()
    pred = []
    for i in range(FOLDS):
        MODEL_FILE = pathlib.Path(__file__).parent.joinpath(f"model{i}.cbm")
        model.load_model(MODEL_FILE)
        predictions = model.predict(features[model.feature_names_])
        pred.append(predictions)
        
    pred = np.mean(pred,axis=0)
    
    preds_df = features[AGG_COLS].copy()
    pred = (features['_vol_tm1']+pred).clip(0,100)
    pred = 10**(pred)-1
    pred = np.floor(pred)
    preds_df["prediction"] = pred
    
    return preds_df
    
