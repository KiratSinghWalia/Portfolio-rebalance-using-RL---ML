load data:
    uv run src/lib/data_script.py

lightgbm:
    uv run src/lib/lightgb.py 



load-data:
    uv run python src/portfolio_rebalance.py load-data \
        --tickers RELIANCE.NS HDFCBANK.NS TCS.NS INFY.NS ICICIBANK.NS HINDUNILVR.NS LT.NS BHARTIARTL.NS ITC.NS ASIANPAINT.NS SBIN.NS GLD TLT \
        --asset_mapping '{"RELIANCE.NS": "equity", "TCS.NS": "equity", "HDFCBANK.NS": "equity", "INFY.NS": "equity", "ICICIBANK.NS": "equity", "BHARTIARTL.NS": "equity", "HINDUNILVR.NS": "equity", "LT.NS": "equity", "ITC.NS": "equity", "ASIANPAINT.NS": "equity", "SBIN.NS": "equity", "GLD": "gold", "TLT": "bond"}'


train-rl-model :
     uv run python src/portfolio_rebalance.py RL-train-model \
        --tickers RELIANCE.NS HDFCBANK.NS TCS.NS INFY.NS ICICIBANK.NS HINDUNILVR.NS LT.NS BHARTIARTL.NS ITC.NS ASIANPAINT.NS SBIN.NS GLD TLT \


rl-compare :
     uv run python src/portfolio_rebalance.py RL-compare \
        --tickers RELIANCE.NS HDFCBANK.NS TCS.NS INFY.NS ICICIBANK.NS HINDUNILVR.NS LT.NS BHARTIARTL.NS ITC.NS ASIANPAINT.NS SBIN.NS GLD TLT \

rl-latest-weights:
    uv run python src/portfolio_rebalance.py RL_latest_weights \
        --tickers RELIANCE.NS HDFCBANK.NS TCS.NS INFY.NS ICICIBANK.NS HINDUNILVR.NS LT.NS BHARTIARTL.NS ITC.NS ASIANPAINT.NS SBIN.NS GLD TLT \


ml-compare:
    uv run python src/portfolio_rebalance.py ML-compare
