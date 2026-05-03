import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from prompt_toolkit.shortcuts import progress_bar
from stable_baselines3 import PPO
from lib.data_script import root_file

class PortfolioEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        X: pd.DataFrame,
        rets: pd.DataFrame,
        tickers,
        max_w=0.35,  #maximum weights 
        turnover_cost_bps=0, #trading cost
        risk_aversion=0.10, 
        lookback_risk=21, #number of days used for risk calculation
    ):
        super().__init__()
        self.X = X
        self.rets = rets.loc[X.index] #aligns returns with same date as X
        self.tickers = list(tickers)
        self.n = len(self.tickers) #total stocks

        self.max_w = max_w
        self.turnover_cost = turnover_cost_bps / 10_000.0 # 1basis point
        self.risk_aversion = risk_aversion
        self.lookback_risk = lookback_risk

        self.dates = list(self.X.index) #storing all dates
        self.t = 0

        obs_dim = self.X.shape[1] + self.n #obs contain two parts market features + previous portfolio weights [feature1, feature2, ..., feature20, previous_weight1, ...,]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32 )
        


        self.action_space = spaces.Box(low=-5, high=5, shape=(self.n,), dtype=np.float32) #a weight for each asset 

        self.w_prev = np.ones(self.n, dtype=np.float32) / self.n #This creates an equal-weight portfolio.

    def reset(self, seed=None, options=None): #restart enviroment 
        super().reset(seed=seed)
        self.t = 0
        self.w_prev = np.ones(self.n, dtype=np.float32) / self.n
        return self._obs(), {}

    def _obs(self): #basically obs for the current date 
        x_t = self.X.iloc[self.t].to_numpy(dtype=np.float32) #Flatenning two level dataframe into one 
        obs = np.concatenate([x_t, self.w_prev]).astype(np.float32) #[momentum,volatility.....weight1,weight2,weight3] , total dims of obs = (features * tickers + number of tickers)
        return obs

    def _softmax(self, a): # similar to pytorch softmax NN sends you actions convert to probabilities 
        a = a - np.max(a)
        e = np.exp(a)
        w = e / (np.sum(e) + 1e-12)
        return w.astype(np.float32)

    def _apply_constraints(self, w): # since we are allocating the softmax as weights , max weights needs to be cosidered 
        # cap weights and renormalize
        w = np.minimum(w, self.max_w)
        s = w.sum()
        if s <= 1e-12:
            return np.ones(self.n, dtype=np.float32) / self.n
        return (w / s).astype(np.float32) #Renormalizes weights so they sum to 1.

    
    def step(self, action):
        done = (self.t >= len(self.dates) - 2) #

        today = self.dates[self.t] 
        next_day = self.dates[self.t + 1]

        w = self._softmax(np.array(action, dtype=np.float32)) #softmax the NN output
        w = self._apply_constraints(w) #re weight it according to constraints 

        r_vec = self.rets.loc[next_day, self.tickers].to_numpy(dtype=np.float32) #Get next day returns

     
        port_r = float(np.dot(w, r_vec)) #Get next day returns

     
        turnover = float(np.sum(np.abs(w - self.w_prev))) #how much weightage we changed 
        cost = self.turnover_cost * turnover #transaction cost for the day 

        t0 = max(0, self.t - self.lookback_risk) #how far back to look for risk calculation., we are using 20 days 
        hist = self.rets.loc[self.dates[t0]:self.dates[self.t]][self.tickers].to_numpy(dtype=np.float32) #past 21 days of returns for selected tickers

        if len(hist) >= 5: #atleast 5 return days 
            port_hist = hist @ w #returns past those days
            risk = float(np.std(port_hist)) #volatility
        else:
            risk = 0.0

        ew = float(np.mean(r_vec)) #Equal-weight benchmark

        reward = (port_r - ew) - cost - self.risk_aversion * risk # reward for changing weights , rather that letting it be as normal 

        self.w_prev = w
        self.t += 1

        obs = self._obs()
        terminated = done
        truncated = False
        info = {
            "date": str(today),
            "port_r": port_r,
            "turnover": turnover,
            "risk": risk,
            "weights": w,
        }
        return obs, reward, terminated, truncated, info
    
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

class RL_model:
    def __init__(self,tickers):
        import yfinance as yf
        import pandas as pd
        self.cut_off = "2025-01-01"
        self.tickers = tickers
        self.prices1  = yf.download(self.tickers, start="2022-01-01", end="2026-04-30", auto_adjust=True, progress=False)["Close"]
        self.prices_before_cutoff = self.prices1.loc[:self.cut_off].copy()
        self.prices_after_cutoff = self.prices1.loc[self.cut_off:].copy()
        self.prices = self.prices_before_cutoff
        self.prices_after_cutoff.index = pd.to_datetime(self.prices_after_cutoff.index)
        self.prices_after_cutoff = self.prices_after_cutoff.sort_index()
        self.prices.index = pd.to_datetime(self.prices.index)
        self.prices = self.prices.sort_index()
        self.prices = self.prices.dropna(how="all")
        self.rets  = self.prices.pct_change().dropna()

    def feature_engineering_and_split(self):
        mom21 = (self.prices / self.prices .shift(21) - 1.0)          # 21-day momentum
        vol21 = self.rets.rolling(21).std()              #21 day volatility 
        mom21 = mom21.reindex(self.rets.index) #align these both to same index that is dates
        vol21 = vol21.reindex(self.rets.index)
        mom5  = (self.prices  / self.prices .shift(5) - 1.0).reindex(self.rets.index) # a short term momentum
        X = pd.concat(
        {"mom5": mom5, "mom21": mom21, "vol21": vol21},
        axis=1
    ).swaplevel(axis=1).sort_index(axis=1) #develope a multiindex dataframe like [ticker->features, n*ticker firstlayer -> features*tickers next]
        X = X.dropna(how="any")
        rets_aligned = self.rets.loc[X.index].copy() # align both dataframes to dates
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        assert X_clean.index.equals(rets_aligned.index)
        split = int(len(X_clean) * 0.8)
        X_train = X_clean.iloc[:split].copy()
        X_test  = X_clean.iloc[split:].copy()   
        rets_train = rets_aligned.iloc[:split].copy()
        rets_test  = rets_aligned.iloc[split:].copy()
        return X_train,X_test,rets_train,rets_test,tickers,X_clean,rets_aligned
    
    def linear_schedule(self,initial_value: float):
        def f(progress_remaining: float):
            # progress_remaining goes 1 -> 0
            return initial_value * progress_remaining
        return f
    
    def weights_proportional(self,pred_by_ticker: pd.Series, top_k=5, max_w=0.35):
        top = pred_by_ticker.sort_values(ascending=False).head(top_k)
        # shift to positive
        x = top - top.min()
        x = x + 1e-6
        w = x / x.sum()
        w = np.minimum(w, max_w)
        w = w / w.sum()

        out = pd.Series(0.0, index=pred_by_ticker.index)
        out.loc[top.index] = w.values
        return out

    def model_training(self):
        X_train,_,rets_train,_,tickers,_,_=self.feature_engineering_and_split()
        train_env = DummyVecEnv([lambda: PortfolioEnv(X_train, rets_train, tickers, max_w=0.35, turnover_cost_bps=5, risk_aversion=0.10)]) #Stable-Baselines3 often expects a vectorized environment.
        model = PPO(
        "MlpPolicy",
        train_env,  
        verbose=1,
        learning_rate=self.linear_schedule(3e-4),
        n_steps=512,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.01,
    )   
        model.learn(total_timesteps=100_000, log_interval=1, progress_bar=False,)
        model.save("ppo_portfolio_model")

    def Monthly_baseline(self):
        monthly_rets = (1.0 + self.rets).resample("ME").prod() - 1.0 #monthly returns compounded
        K = 5
        dates = monthly_rets.index
        weights = pd.DataFrame(0.0, index=dates, columns=monthly_rets.columns) #Create an empty weights table , to store weights to buy from next month
        eps = 1e-12
        for i in range(1, len(dates)):
            prev_month = dates[i-1]
            next_month = dates[i]

        r_prev = monthly_rets.loc[prev_month] #prev month ticker wise returns

        top = r_prev.nlargest(K).index #choose the top 3 
        scores = r_prev[top].clip(lower=0.0)  # only reward positive momentum; optional , Make weights based on momentum strength

    # if all selected scores are 0 (e.g., all negative), fallback to equal weight
        if scores.sum() <= eps:
            w = pd.Series(1.0 / K, index=top)
        else:
            w = scores / scores.sum()

        weights.loc[next_month, top] = w.values # storing weights for next month , and top picks

        port_monthly = (weights * monthly_rets).sum(axis=1) #Calculate portfolio monthly return
        equity = (1.0 + port_monthly.fillna(0.0)).cumprod() # portfolio value over time.


    def latest_portfolio_weights(self):
        _,_,_,_,tickers,X_clean,rets_aligned=self.feature_engineering_and_split()
        env = PortfolioEnv(X_clean, rets_aligned, tickers,
                   max_w=0.30, turnover_cost_bps=5, risk_aversion=0.10) #Creates a fresh environment using all  cleaned data.
        obs, _ = env.reset()
        done = False
        model = PPO.load("ppo_portfolio_model")
        while not done:
            action, _ = model.predict(obs, deterministic=True) #Your trained PPO model looks at the current observation and chooses an action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated #If terminated is True OR truncated is True, then done becomes True.
        w_next = pd.Series(info["weights"], index=tickers)
        print("Sum of weights:", w_next.sum())
        print("Min/Max weight:", w_next.min(), w_next.max())
        print(w_next)
        return w_next

    def comparison(self):
        import plotly.graph_objects as go
        K = 5
        rebalance_every = 5
        max_w = 0.30
        turnover_cost_bps = 5
        risk_aversion = 0.10
        tickers = self.tickers
        prices_after_cutoff = self.prices_after_cutoff.apply(pd.to_numeric, errors="coerce")
        prices_after_cutoff = self.prices_after_cutoff.ffill().dropna(how="all")
        rets_after_cutoff = prices_after_cutoff.pct_change().dropna()
        rets_after_cutoff = rets_after_cutoff.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        model = PPO.load("ppo_portfolio_model")
        #------------------------------------------building same features as before------------------------------------------------------#
        mom5 = (prices_after_cutoff / prices_after_cutoff.shift(5) - 1.0)
        mom21 = (prices_after_cutoff / prices_after_cutoff.shift(21) - 1.0)
        vol21 = rets_after_cutoff.rolling(21).std()

        mom5 = mom5.reindex(rets_after_cutoff.index)
        mom21 = mom21.reindex(rets_after_cutoff.index)
        vol21 = vol21.reindex(rets_after_cutoff.index)

        X_after = pd.concat(
            {"mom5": mom5, "mom21": mom21, "vol21": vol21},
            axis=1
        ).swaplevel(axis=1).sort_index(axis=1)

        X_after = X_after.dropna(how="any")

        rets_after_aligned = rets_after_cutoff.loc[X_after.index].copy()

        X_after_clean = X_after.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        assert X_after_clean.index.equals(rets_after_aligned.index)

        print("X_after_clean:", X_after_clean.shape)
        print("rets_after_aligned:", rets_after_aligned.shape)
         #------------------------------------------building same features as before------------------------------------------------------#
        dates = list(X_after_clean.index)

        momentum_returns = []
        ppo_returns = []
        result_dates = []

        momentum_weight_records = []
        ppo_weight_records = []

        momentum_w = pd.Series(1.0 / len(tickers), index=tickers) #initial port weights are equal
        ppo_w = pd.Series(1.0 / len(tickers), index=tickers)
        
        for t in range(len(dates) - 1):
            today = dates[t]
            next_day = dates[t + 1]

            if t % rebalance_every == 0: #after every 5 days
                mom_signal = mom5.loc[today, tickers]
                momentum_w = momentum_top_k_weights(mom_signal, k=K)
                X_t = X_after_clean.loc[today].to_numpy(dtype=np.float32)

                # PPO observation = features + previous PPO weights
                obs = np.concatenate(
                    [X_t, ppo_w.values.astype(np.float32)]
                ).astype(np.float32)

                action, _ = model.predict(obs, deterministic=True)

                ppo_w_full = softmax_to_weights(action)
                ppo_w_full = apply_simple_max_weight(ppo_w_full, max_w=max_w)

                # Keep only top 5 PPO weights
                ppo_w = keep_top_k(ppo_w_full, k=K)

            r_next = rets_after_aligned.loc[next_day, tickers]
            momentum_r = float(np.dot(momentum_w.values, r_next.values))
            ppo_r = float(np.dot(ppo_w.values, r_next.values))

            momentum_returns.append(momentum_r)
            ppo_returns.append(ppo_r)
            result_dates.append(next_day)

            momentum_weight_records.append(momentum_w.rename(next_day))
            ppo_weight_records.append(ppo_w.rename(next_day))
 
        momentum_returns = pd.Series(momentum_returns, index=pd.to_datetime(result_dates))
        ppo_returns = pd.Series(ppo_returns, index=pd.to_datetime(result_dates))

        momentum_equity_5d = (1.0 + momentum_returns).cumprod()
        ppo_equity_5d = (1.0 + ppo_returns).cumprod()

        momentum_weights_5d = pd.DataFrame(momentum_weight_records)
        ppo_weights_5d = pd.DataFrame(ppo_weight_records)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(11, 5))

        plt.plot(
            momentum_equity_5d.index,
            momentum_equity_5d.values,
            marker="o",
            label=f"Momentum Top {K} - Rebalanced Every {rebalance_every} Days"
        )

        plt.plot(
            ppo_equity_5d.index,
            ppo_equity_5d.values,
            marker="o",
            label=f"PPO/RL Top {K} - Rebalanced Every {rebalance_every} Days"
        )

        plt.title("Equity Curve Comparison: $1 Invested After Cutoff")
        plt.xlabel("Date")
        plt.ylabel("Equity Growth of $1")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show()


#-----------------------helper fuctions as PPO output raw values---------------------------------#

def softmax_to_weights(action):
    action = np.array(action, dtype=np.float32)
    action = action - np.max(action)
    e = np.exp(action)
    w = e / (np.sum(e) + 1e-12)
    return pd.Series(w, index=tickers)


def apply_simple_max_weight(w, max_w=0.35):
    w = w.clip(upper=max_w)
    if w.sum() <= 1e-12:
        return pd.Series(1.0 / len(w), index=w.index)
    return w / w.sum()

def keep_top_k(w, k=5):
    top = w.nlargest(k)
    out = pd.Series(0.0, index=w.index)
    out.loc[top.index] = top.values

    if out.sum() <= 1e-12:
        out.loc[top.index] = 1.0 / k
    else:
        out = out / out.sum()

    return out


def momentum_top_k_weights(signal_row, k=5):
    top = signal_row.nlargest(k)
    scores = top.clip(lower=0.0)

    if scores.sum() <= 1e-12:
        w = pd.Series(1.0 / k, index=top.index)
    else:
        w = scores / scores.sum()

    out = pd.Series(0.0, index=signal_row.index)
    out.loc[w.index] = w.values

    return out

if __name__ == "__main__":
    pass



