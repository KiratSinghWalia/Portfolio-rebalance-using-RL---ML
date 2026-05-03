# Portfolio Rebalancing CLI using Reinforcement Learning & Machine Learning

> **Disclaimer** — This model is a small project trained on a limited number of stocks. Do not make any real investment decisions based on it.

---

## 🤖 RL Model (PPO)

The PPO (Proximal Policy Optimization) agent observes each ticker's:
- 5-day momentum
- 21-day momentum
- 21-day volatility
- Previous portfolio weights

It outputs allocation scores which are converted into portfolio weights, keeping only the **top K tickers**.

In testing, the model rebalances every **N days** (e.g. every 5 days), holds those weights between rebalances, and compares the $1 equity curve against a **momentum strategy**.

---

## 📊 LightGBM Model (ML)

The LightGBM model takes a supervised ML approach to portfolio allocation:
- Trained on engineered features including 1-day, 5-day, and 20-day returns, 20-day volatility, z-score, price-to-MA20, volume ratio, and volume change
- Predicts **future 5-day returns** for each asset
- Ranks assets by predicted return and allocates weights accordingly
- Compared against an **equal-weight benchmark**

---

## 📦 Assets Covered

| Ticker | Type |
|---|---|
| RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS | Equity |
| BHARTIARTL.NS, HINDUNILVR.NS, LT.NS, ITC.NS, ASIANPAINT.NS, SBIN.NS | Equity |
| GLD | Gold |
| TLT | Bond |

---

## 🚀 Setup

### Prerequisites
- [uv](https://docs.astral.sh/uv/) — Python package manager
- [just](https://just.systems) — task runner

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install just
```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
```

### Clone & Install
```bash
git clone https://github.com/KiratSinghWalia/Portfolio-rebalance-using-RL---ML.git
cd Portfolio-rebalance-using-RL---ML
uv sync
```

> This creates a `.venv` inside the project. Your global Python is untouched.

---

## 🛠 Usage



### Step 1 — Train RL Model

```bash
just train-rl-model
```

Or:
```bash
uv run python src/portfolio_rebalance.py RL-train-model \
    --tickers RELIANCE.NS HDFCBANK.NS TCS.NS INFY.NS ICICIBANK.NS HINDUNILVR.NS LT.NS BHARTIARTL.NS ITC.NS ASIANPAINT.NS SBIN.NS GLD TLT
```

> ⚠️ If you want to change tickers, use the CLI command directly — not `just`.

---

### Step 2 — Compare RL vs Momentum Strategy

```bash
just rl-compare
```

Or:
```bash
uv run python src/portfolio_rebalance.py RL-compare \
    --tickers RELIANCE.NS HDFCBANK.NS TCS.NS INFY.NS ICICIBANK.NS HINDUNILVR.NS LT.NS BHARTIARTL.NS ITC.NS ASIANPAINT.NS SBIN.NS GLD TLT
```

<img width="1103" height="495" alt="image" src="https://github.com/user-attachments/assets/a4b954f0-13ed-44a1-8872-f6ba79dddf31" />


---

### Get Latest RL Portfolio Weights

```bash
just rl-latest-weights
```
> ⚠️ If you want to inference
---

## Using ML Model 
### Step 1 — Load Data

```bash
just load-data
```

Or with custom tickers (use CLI directly):
```bash
uv run python src/portfolio_rebalance.py load-data \
    --tickers RELIANCE.NS HDFCBANK.NS TCS.NS INFY.NS ICICIBANK.NS HINDUNILVR.NS LT.NS BHARTIARTL.NS ITC.NS ASIANPAINT.NS SBIN.NS GLD TLT \
    --asset_mapping '{"RELIANCE.NS": "equity", "TCS.NS": "equity", "HDFCBANK.NS": "equity", "INFY.NS": "equity", "ICICIBANK.NS": "equity", "BHARTIARTL.NS": "equity", "HINDUNILVR.NS": "equity", "LT.NS": "equity", "ITC.NS": "equity", "ASIANPAINT.NS": "equity", "SBIN.NS": "equity", "GLD": "gold", "TLT": "bond"}'
```

> ⚠️ If you want to change tickers, use the CLI command directly — not `just`.

---
### Step 2 — Train & Compare LightGBM Model

Compare ML vs equal-weight benchmark:
```bash
just ml-compare
```

Or:
```bash
uv run python src/portfolio_rebalance.py ML-compare
```
<img width="1199" height="697" alt="image" src="https://github.com/user-attachments/assets/0da12007-a31e-4cc7-a205-78a1c64cc91a" />

---

## 📁 Project Structure

```
.
├── src/
│   ├── portfolio_rebalance.py   # Main CLI entry point
│   └── lib/
│       ├── data_script.py       # Data downloading & feature engineering
│       ├── lightgb.py           # LightGBM ML model
│       └── RL_script.py         # RL training & evaluation
├── data/                        # Generated data (gitignored)
├── .devcontainer/               # Dev container config
├── justfile                     # Task runner commands
├── pyproject.toml               # Project dependencies
└── uv.lock                      # Locked dependency versions
```

---

## 🔧 Tech Stack

- **[uv](https://docs.astral.sh/uv/)** — dependency management
- **[Stable Baselines3](https://stable-baselines3.readthedocs.io/)** — RL training (PPO)
- **[Gymnasium](https://gymnasium.farama.org/)** — RL environment
- **[LightGBM](https://lightgbm.readthedocs.io/)** — ML model
- **[yFinance](https://pypi.org/project/yfinance/)** — market data
- **[Pandas](https://pandas.pydata.org/)** / **[NumPy](https://numpy.org/)** — data processing
- **[Plotly](https://plotly.com/python/)** / **[Matplotlib](https://matplotlib.org/)** — visualisation
