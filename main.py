import pandas as pd

from engine.strategy_loader import (
    load_strategy
)

from engine.vectorbt_runner import (
    VectorBTRunner
)

from engine.data_manager import (
    DataManager
)


def log(msg):
    print(msg)


if __name__ == "__main__":

    strategy = load_strategy(
        "strategies/kd_basic.py"
    )

    manager = DataManager(
        csv_folder="data",
        logger=log
    )

    datasets = {}

    ticker = "2330.TW"

    df = manager.load(
        ticker
    )

    if df is None:
        raise RuntimeError(
            "下載失敗"
        )

    datasets[ticker] = df

    strategy_config = {
        "period": [9],
        "alpha_k": [0.333],
        "alpha_d": [0.333],
        "oversold": [20],
        "overbought": [80]
    }

    runner = VectorBTRunner(
        strategy=strategy,
        logger=log
    )

    results = runner.run(
        datasets=datasets,
        strategy_config=strategy_config
    )

    print()

    print("=" * 60)

    print(
        f"Results = {len(results)}"
    )

    if results:
        print(
            results[0]
        )

    print("=" * 60)