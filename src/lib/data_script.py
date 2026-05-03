from pathlib import Path
import numpy as np
from sklearn.preprocessing import OneHotEncoder
root_file = Path(__file__).parent.parent.parent
data_path = root_file/'data'/'data.csv'


def Data(tickers,asset_type):
    root_file = Path(__file__).parent.parent.parent
    data_path = root_file/'data'
    import yfinance as yf
    import pandas as pd
    data = yf.download(tickers, start= "2022-01-01", group_by="ticker")
    dfs = []
    for ticker in tickers:
        df = data[ticker].copy()
        df["asset"] = ticker
        df = df.reset_index()
        dfs.append(df)

    data = pd.concat(dfs)
    data.columns = [c.lower() for c in data.columns]

    data = data[["date","asset","open","high","low","close","volume"]]
    data = data.sort_values(["asset","date"]).dropna()
    data = feature_enginering(data)
    data = data.dropna()
    data.to_csv(data_path/'data.csv',index=False)
    return data

def feature_enginering(data):
    import pandas as pd

    data["ret_1d"]  = data.groupby("asset")["close"].pct_change(1)
    data["ret_5d"]  = data.groupby("asset")["close"].pct_change(5)
    data["ret_20d"] = data.groupby("asset")["close"].pct_change(20)

    data["vol_20d"] = (
    data.groupby("asset")["ret_1d"]
    .rolling(20).std()
    .reset_index(level=0, drop=True)
)
    data["ma_20"] = (
    data.groupby("asset")["close"]
    .rolling(20).mean()
    .reset_index(level=0, drop=True)
)

    data["price_to_ma20"] = data["close"] / data["ma_20"]

    rolling_mean = data.groupby("asset")["close"].rolling(20).mean()
    rolling_std  = data.groupby("asset")["close"].rolling(20).std()

    data["zscore_20"] = (
    (data["close"] - rolling_mean.reset_index(level=0, drop=True)) /
    rolling_std.reset_index(level=0, drop=True)
)

    data["rank_ret_20d"] = data.groupby("date")["ret_20d"].rank()


    data["vol_ma20"] = (
    data.groupby("asset")["volume"]
    .rolling(20).mean()
    .reset_index(level=0, drop=True)
)

    data["vol_ratio"] = data["volume"] / data["vol_ma20"]


    data["ret_5d_vol"] = data["ret_5d"] * data["vol_ratio"]     

    data["vol_chg_5d"] = data.groupby("asset")["volume"].pct_change(5)
    data["vol_chg_5d"] = data["vol_chg_5d"].replace([np.inf, -np.inf], np.nan)
    data["vol_chg_5d"] = data["vol_chg_5d"].clip(-5, 5) #reducing noise

    data["future_ret_5d"]= data.groupby("asset")["close"].pct_change(5).shift(-5)
    data["asset_type"] = data["asset"].map(asset_type_map)
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(data[['asset_type']])

    encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(['asset_type'])
)
    data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)
    data = data.drop(columns=["asset_type"])
    #data['ret5_x_vol'] = data['ret_5d'] * data['vol_20d']
    #data['momentum_divergence'] = data['ret_1d'] - data['ret_20d']
    
    data = data.drop(columns = ['high','low','open','close','volume','ma_20','vol_ma20','ret_5d_vol'])
    data = data.dropna()
    return data


tickers = ["RELIANCE.NS",    
 "HDFCBANK.NS",    
 "TCS.NS",         
 "INFY.NS",        
 "ICICIBANK.NS",   
 "HINDUNILVR.NS",  
 "LT.NS",          
 "BHARTIARTL.NS",  
 "ITC.NS",         
 "ASIANPAINT.NS",  
 "SBIN.NS",        
 "GLD",            
 "TLT"]            

asset_type_map = {
    "RELIANCE.NS": "equity",
    "TCS.NS": "equity", 
    "HDFCBANK.NS": "equity",
    "INFY.NS": "equity",
    "ICICIBANK.NS": "equity",
    "BHARTIARTL.NS": "equity",
    "HINDUNILVR.NS": "equity",
    "LT.NS": "equity",
    "ITC.NS": "equity",
    "ASIANPAINT.NS": "equity",
    "SBIN.NS": "equity",
    "GLD": "gold",
    "TLT": "bond"}


def data_RL(ticker,start:str,end:str):
    root_file = Path(__file__).parent.parent.parent
    data_path = root_file/'data'
    import yfinance as yf
    import pandas as pd
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(how="all")
    rets = prices.pct_change().dropna()
    prices.to_csv(data_path/'prices.csv',index=True)
    rets.to_csv(data_path/'rets.csv',index=True)

if __name__ == "__main__":
    pass

