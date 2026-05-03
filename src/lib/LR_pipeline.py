
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
class Linear_Pipeline:
    def __init__(self,data,N_SPLITS:int,MIN_TRAIN_MONTHS:int):
        self.data = data
        self.features,self.Target = self.Target_features(self.data)
        self.Nsplits = N_SPLITS
        self.Min_train_months = MIN_TRAIN_MONTHS
        self.ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]

    def Target_features(self,data):
        REMOVE = {"date","asset","asset_type","future_ret_5d","price_to_ma20"}#"vol_20d","vol_ratio","vol_chg_5d"}
        FEATURES = [i for i in data.columns.tolist() if i not in REMOVE]
        Target = "future_ret_5d"
        return FEATURES,Target

    def walk_forward_splits (self,data, n_splits: int, min_train_months: int):
        data["date"] = pd.to_datetime(data["date"])
        dates = data["date"].sort_values().unique() #index of all unique dates
        cutoff = pd.Timestamp(dates[0]) + pd.DateOffset(months=min_train_months) #cutoff according to our min train months
        available = dates[dates >= cutoff] #total available dates for val after first training.

        fold_edges = np.array_split(available, n_splits) # splitting all aur 

        splits = []

        for fold in fold_edges:
            val_start = fold[0]
            val_end   = fold[-1]
            train_idx = data.index[data["date"] < val_start]
            val_idx   = data.index[(data["date"] >= val_start) & (data["date"] <= val_end)]
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
        return splits

    def IC(self,y_true,y_pred): # cuz we are working with financial data will will use rank spearmanr.
        ic, _ = spearmanr(y_true, y_pred)
        return ic if not np.isnan(ic) else 0.

    def hit_rate(self,y_true, y_pred) -> float:
        return np.mean(np.sign(y_true) == np.sign(y_pred)) #Avg of predictions with correct sign

    def compute_metrics(self,y_true, y_pred, label="") -> dict:
        ic   = self.IC(y_true, y_pred)
        hr   = self.hit_rate(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        if label:
            print(f"  [{label}]  IC={ic:.4f}  Hit={hr:.3f}  RMSE={rmse:.6f}")
        return {"ic": ic, "hit_rate": hr, "rmse": rmse}

    def run_walk_forward(self,data:pd.DataFrame,model_name:str,alpha=100.0):
        splits = self.walk_forward_splits(self.data, self.Nsplits, self.Min_train_months)
        if model_name == "linear":
            base_model = LinearRegression()
        elif model_name == "ridge":
            base_model = Ridge(alpha=alpha)
        elif model_name == "lasso":
            base_model = Lasso(alpha=alpha, max_iter=5000)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
        model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", base_model),
        ])

        fold_results = []
        all_preds    = []

        for i, (train_idx, val_idx) in enumerate(splits):
            X_train = data.loc[train_idx, self.features].values
            y_train = data.loc[train_idx, self.Target].values
            X_val   = data.loc[val_idx,   self.features].values
            y_val   = data.loc[val_idx,   self.Target].values
        
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            metrics = self.compute_metrics(y_val, y_pred, label=f"fold {i+1}")
            metrics["fold"] = i + 1
            metrics["n_train"] = len(train_idx)
            metrics["n_val"]   = len(val_idx)
            fold_results.append(metrics)

            pred_df = data.loc[val_idx, ["date", "asset", self.Target]].copy()
            pred_df["y_pred"] = y_pred
            pred_df["fold"]   = i + 1
            all_preds.append(pred_df)

        results_df = pd.DataFrame(fold_results)
        print(f"\n  Mean IC   : {results_df['ic'].mean():.4f} ± {results_df['ic'].std():.4f}")
        print(f"  Mean Hit  : {results_df['hit_rate'].mean():.3f}")
        print(f"  Mean RMSE : {results_df['rmse'].mean():.6f}")
 
        preds_df = pd.concat(all_preds, ignore_index=True)
        return results_df, preds_df, model

    def select_alpha(self,data,model_name):
        print(f"\n── Alpha search for {model_name} ──")
        best_alpha, best_ic = None, -np.inf
        for alpha in self.ALPHA_GRID:
            pass
        pass

if __name__ == "__main__":
    pass