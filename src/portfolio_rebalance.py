import argparse

import json



def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio Rebalancing CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    load_data = subparsers.add_parser("load-data", help="load data according to the given ticker")
    load_data.add_argument("--tickers", type=str, nargs="+", help="List of tickers to load data")
    load_data.add_argument("--asset_mapping", type= json.loads, help="dict of assets mapping to its class")

    RLtrainmodel = subparsers.add_parser("RL-train-model", help="Use RL to optimize porfolio")
    RLtrainmodel.add_argument("--tickers", type=str, nargs="+", help="List of tickers to load data")

    RL_comparison = subparsers.add_parser("RL-compare", help="Use RL to optimize porfolio and compare it with momentum based returns")
    RL_comparison.add_argument("--tickers", type=str, nargs="+", help="List of tickers to load data")

    RL_latest_weights= subparsers.add_parser("RL_latest_weights", help="Use RL to optimize porfolio and compare it with momentum based returns")
    RL_latest_weights.add_argument("--tickers", type=str, nargs="+", help="List of tickers to load data")
    
    ML_comparison = subparsers.add_parser("ML-compare", help="Use ML to optimize porfolio and compare it with momentum based returns")


    args = parser.parse_args()

    match args.command:
        case "load-data":
            from lib.data_script import Data
            Data(args.tickers,args.asset_mapping)

        case "RL-train-model":
            from lib.RL_script import RL_model
            RL = RL_model(args.tickers)
            RL.model_training()
        
        case "RL-compare":
            from lib.RL_script import RL_model
            RL = RL_model(args.tickers)
            RL.comparison()
        
        case "ML-compare":
            import subprocess
            import sys
            from pathlib import Path
            script = Path(__file__).parent / 'lib' / 'lightgb.py'
            subprocess.run([sys.executable, str(script)], check=True)
        case "RL_latest_weights":
            from lib.RL_script import RL_model
            RL = RL_model(args.tickers)
            RL.latest_portfolio_weights()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()