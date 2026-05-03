
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import lightgbm as lgb
from datetime import datetime
from pathlib import Path
try:
    from data_script import data_path
except ImportError:
    from lib.data_script import data_path
data = pd.read_csv(data_path)

class Lightgbm:
    def __init__(self,data):
        self.data = data
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.train_data = self.data[self.data['date'] < '2025-01-01']
        self.test_data  = self.data[self.data['date'] >= '2025-01-01']
        #split_point = int(len(self.data1) * 0.7)
        #self.data = self.data1.iloc[:split_point]
        self.features,self.Target = self.Target_features(self.data)
        self.Nsplits = 5
        self.Min_train_months = 6
        self.params = {
    'objective':         'regression',
    'metric':            'rmse',
    'num_leaves':        31,
    'learning_rate':     0.05,
    'n_estimators':      200,
    'min_child_samples': 20,
    'verbosity':         -1
}

    def Target_features(self,data):
        REMOVE = {"date","asset","asset_type","future_ret_5d","price_to_ma20"}#"vol_20d","vol_ratio","vol_chg_5d"}
        FEATURES = [i for i in data.columns.tolist() if i not in REMOVE]
        TARGET = "future_ret_5d"
        return FEATURES,TARGET
    
    def Total_data(self):
        data = self.data
        data = data.sort_values('date').reset_index(drop =True) 
        split = int(len(data) * 0.8)
        train = data.iloc[:split]
        val   = data.iloc[split:]
        X_train, y_train = train[self.features].values, train[self.Target].values
        X_val,   y_val   = val[self.features].values,   val[self.Target].values
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data   = lgb.Dataset(X_val,   label=y_val)

        Model = lgb.train(
    self.params,
    train_data,
    valid_sets=[val_data],
)
        y_pred = Model.predict(X_val)

        ic, _ = spearmanr(y_val, y_pred)
        hit   = np.mean(np.sign(y_val) == np.sign(y_pred))

        print(f"IC  : {ic:.4f}")
        print(f"Hit : {hit:.3f}")
    
        return ic,hit
    
    def walk_forward_lgb(self):
        import joblib
        data = self.train_data
        data["date"] = pd.to_datetime(data["date"])
        dates = data['date'].sort_values().unique()
        cutoff    = pd.Timestamp(dates[0]) + pd.DateOffset(months=self.Min_train_months)
        available = dates[dates >= cutoff]
        folds  = np.array_split(available, self.Nsplits)
        all_preds = [] 

        for i ,fold in enumerate(folds):
            val_start = fold[0]
            val_end = fold[-1]

            train = data[data['date'] < val_start - pd.DateOffset(days=5)] #-5 cuz we are predicting next 5 day returns
            val   = data[(data['date'] >= val_start) & (data['date'] <= val_end)].copy()

            X_tr, y_tr = train[self.features].values, train[self.Target].values
            X_vl, y_vl = val[self.features].values,   val[self.Target].values

            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data   = lgb.Dataset(X_vl, label=y_vl)

            model = lgb.train(self.params, train_data, valid_sets=[val_data])
            val['predicted'] = model.predict(X_vl)   
            val['fold']      = i + 1
            y_pred = model.predict(X_vl)
            ic, _  = spearmanr(y_vl, y_pred)
            hit    = np.mean(np.sign(y_vl) == np.sign(y_pred))

            print(f"Fold {i+1}  IC={ic:.4f}  Hit={hit:.3f}")
            all_preds.append(val[['date', 'asset', self.Target, 'predicted', 'vol_20d', 'fold']])
        
        preds = pd.concat(all_preds).reset_index(drop=True)
        preds['rank'] = preds.groupby('date')['predicted'].rank(ascending=False).astype(int)
        preds['rank_score'] = preds.groupby('date')['rank'].transform(lambda r: r.max() - r + 1)
        preds['score'] = preds['rank_score'] * (1 / preds['vol_20d'])
        rootfile = Path(__file__).parent.parent.parent
        #self.model = model
        return preds, model 
    
    def final_output(self):
        total_data_ic, total_data_hit = self.Total_data()
        preds, self.model = self.walk_forward_lgb()
        return preds
    
    def comparison(self):
        rootfile   = Path(__file__).parent.parent.parent
        model_path = rootfile / 'lgbm_final_model.txt'

        if hasattr(self, 'model'):
            model = self.model  # use already trained model
            print("Using in-memory model")
        elif model_path.exists():
            model = lgb.Booster(model_file=str(model_path))
            print(f"Loaded model from {model_path}")
        else:
            print("No model found — training now...")
        self.final_output()
        model = self.model
        #model = self.walk_forward_lgb()
        test_data = self.test_data
        test_data['predicted'] = model.predict(test_data[self.features].values)   
        ic, _ = spearmanr(test_data[self.Target], test_data['predicted'])
        hit   = np.mean(np.sign(test_data[self.Target]) == np.sign(test_data['predicted']))
        print(f"Test IC  : {ic:.4f}")
        print(f"Test Hit : {hit:.3f}")    
        test_data['ml_rank']  = test_data.groupby('date')['predicted'].rank(ascending=False).astype(int)
        test_data['mom_rank'] = test_data.groupby('date')['ret_5d'].rank(ascending=False).astype(int)
        INITIAL_CAPITAL = 1000
        rebalance_dates = np.sort( #dates on which model gets rebalanced [every 5 day]
            test_data[test_data['asset_type_equity']==1.0]['date'].unique()
        )[::5]

        def sharpe(rets):   return (rets.mean()/rets.std()) * np.sqrt(252/5)
        def max_drawdown(v): return ((v - v.cummax())/v.cummax()).min()

        records = {'ml_top5': [], 'mom_top5': []}

        for date in rebalance_dates:
            day = test_data[test_data['date'] == date].copy()
            eq  = day[day['asset_type_equity'] == 1.0].copy()

            # ML top 5 — equal weight
            top5_ml           = eq.nlargest(5, 'predicted').copy()
            top5_ml['weight'] = 1/5
            records['ml_top5'].append({
                'date': date,
                'ret' : (top5_ml['weight'] * top5_ml[self.Target]).sum(),
                'assets': top5_ml['asset'].tolist()
            })

            # Momentum top 5 — equal weight
            top5_mom           = eq.nlargest(5, 'ret_5d').copy()
            top5_mom['weight'] = 1/5
            records['mom_top5'].append({
                'date': date,
                'ret' : (top5_mom['weight'] * top5_mom[self.Target]).sum(),
                'assets': top5_mom['asset'].tolist()
            })

        dfs = {k: pd.DataFrame(v).set_index('date') for k, v in records.items()}
        for k in dfs:
            dfs[k]['value'] = INITIAL_CAPITAL * (1 + dfs[k]['ret']).cumprod()

        labels = {'ml_top5': 'ML top 5', 'mom_top5': 'Momentum top 5'}
        colors = {'ml_top5': '#1D9E75', 'mom_top5': '#185FA5'}

        print(f"\n{'':20} {'Sharpe':>8} {'Ann.Ret':>8} {'Max DD':>8} {'Final $':>10}")
        print('-' * 62)
        for k, label in labels.items():
            df = dfs[k]
            print(f"{label:20} {sharpe(df['ret']):>8.3f} "
                f"{df['ret'].mean()*252/5:>8.3f} "
                f"{max_drawdown(df['value']):>8.3f} "
                f"${df['value'].iloc[-1]:>9.2f}")
            
        fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                         gridspec_kw={'height_ratios': [3, 1]})

        for k, label in labels.items():
            df = dfs[k]
            axes[0].plot(df.index, df['value'], label=label,
                        color=colors[k], linewidth=2)
            dd = (df['value'] - df['value'].cummax()) / df['value'].cummax()
            axes[1].fill_between(dd.index, dd.values, 0,
                                alpha=0.4, color=colors[k], label=label)

        axes[0].axhline(INITIAL_CAPITAL, color='#aaa', linewidth=0.8, linestyle='--')
        axes[0].set_title('$1,000 — ML vs Momentum top 5 (2025–2026 unseen)', fontsize=13)
        axes[0].set_ylabel('Portfolio value ($)')
        axes[0].legend()

        axes[1].set_ylabel('Drawdown')
        axes[1].set_title('Drawdown', fontsize=11)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('test_comparison.png', dpi=150)
        plt.show()
        print("Plot saved → test_comparison.png")



def compare(data):
    from pathlib import Path
    rootfile   = Path(__file__).parent.parent.parent
    model_path = rootfile / 'lgbm_final_model.txt'
    print(f"Looking for model at: {model_path}")
    ML = Lightgbm(data)
    ML.comparison()


if __name__ == '__main__':
    compare(data)


    
