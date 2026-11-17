import itertools
import time

import pandas as pd
import vectorbt as vbt

from engine.result_ranker import (
    ResultRanker
)


class VectorBTRunner:

    def __init__(
        self,
        strategy,
        logger=None
    ):
        self.strategy = strategy
        self.logger = logger

        self.indicator_cache = {}

    def log(self, text):

        if self.logger:
            self.logger(text)

    def build_param_grid(
        self,
        strategy_config
    ):

        names = list(
            strategy_config.keys()
        )

        values = list(
            strategy_config.values()
        )

        rows = []

        for combo in itertools.product(
            *values
        ):

            rows.append(
                dict(
                    zip(
                        names,
                        combo
                    )
                )
            )

        return rows

    def get_indicator_cache(
        self,
        ticker,
        df,
        params
    ):

        cache_key = (
            ticker,
            params["period"],
            params["alpha_k"],
            params["alpha_d"]
        )

        if cache_key not in self.indicator_cache:

            indicators = (
                self.strategy
                .generate_indicators(
                    df=df,
                    period=params["period"],
                    alpha_k=params["alpha_k"],
                    alpha_d=params["alpha_d"]
                )
            )

            self.indicator_cache[
                cache_key
            ] = indicators

        return self.indicator_cache[
            cache_key
        ]

    def run_symbol(
        self,
        ticker,
        df,
        strategy_config
    ):

        ranker = ResultRanker(
            top_n=100
        )

        param_grid = (
            self.build_param_grid(
                strategy_config
            )
        )

        total = len(
            param_grid
        )

        self.log(
            f"[{ticker}] "
            f"參數組數={total}"
        )

        close = df["Close"]

        start_time = (
            time.perf_counter()
        )

        for idx, params in enumerate(
            param_grid,
            start=1
        ):

            try:

                indicators = (
                    self.get_indicator_cache(
                        ticker=ticker,
                        df=df,
                        params=params
                    )
                )

                k = indicators["K"]
                d = indicators["D"]

                entries = (
                    (k > d)
                    &
                    (
                        k
                        <
                        params["oversold"]
                    )
                )

                exits = (
                    (k < d)
                    &
                    (
                        k
                        >
                        params[
                            "overbought"
                        ]
                    )
                )

                pf = (
                    vbt.Portfolio
                    .from_signals(
                        close,
                        entries,
                        exits,
                        freq="1D"
                    )
                )

                try:
                    total_return = (
                        float(
                            pf.total_return()
                        )
                        * 100
                    )
                except Exception:
                    total_return = 0

                try:
                    sharpe = float(
                        pf.sharpe_ratio()
                    )
                except Exception:
                    sharpe = 0

                try:
                    maxdd = (
                        float(
                            pf.max_drawdown()
                        )
                        * 100
                    )
                except Exception:
                    maxdd = 0

                ranker.add(
                    {
                        "ticker": ticker,
                        "return": total_return,
                        "sharpe": sharpe,
                        "maxdd": maxdd,
                        **params
                    }
                )

            except Exception as e:

                self.log(
                    f"[ERROR] "
                    f"{ticker} "
                    f"{params} "
                    f"{str(e)}"
                )

            if idx % 500 == 0:

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                speed = (
                    idx
                    /
                    max(
                        elapsed,
                        0.0001
                    )
                )

                self.log(
                    f"[{ticker}] "
                    f"{idx:,}/{total:,} "
                    f"| "
                    f"{speed:.2f} "
                    f"tests/sec"
                )

        return ranker.get_results()

    def run(
        self,
        datasets,
        strategy_config
    ):

        all_results = []

        global_start = (
            time.perf_counter()
        )

        for ticker, df in datasets.items():

            rows = self.run_symbol(
                ticker=ticker,
                df=df,
                strategy_config=strategy_config
            )

            all_results.extend(
                rows
            )

        elapsed = (
            time.perf_counter()
            - global_start
        )

        self.log("")
        self.log(
            "=" * 60
        )

        self.log(
            f"總耗時 "
            f"{elapsed:.2f} 秒"
        )

        self.log(
            f"總結果數 "
            f"{len(all_results):,}"
        )

        self.log(
            "=" * 60
        )

        return all_results