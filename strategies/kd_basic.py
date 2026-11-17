import pandas as pd


class Strategy:

    NAME = "KD 基礎策略"

    MATRIX_SUPPORT = True

    PARAMETERS = [
        "period",
        "alpha_k",
        "alpha_d",
        "oversold",
        "overbought"
    ]

    def generate_signals(
        self,
        df,
        period,
        alpha_k,
        alpha_d,
        oversold,
        overbought
    ):

        low_min = (
            df["Low"]
            .rolling(window=period)
            .min()
        )

        high_max = (
            df["High"]
            .rolling(window=period)
            .max()
        )

        rsv = (
            (
                df["Close"]
                - low_min
            )
            /
            (
                high_max
                - low_min
                + 1e-9
            )
        ) * 100

        k = (
            rsv
            .ewm(
                alpha=alpha_k,
                adjust=False
            )
            .mean()
        )

        d = (
            k
            .ewm(
                alpha=alpha_d,
                adjust=False
            )
            .mean()
        )

        entries = (
            (k > d)
            &
            (k < oversold)
        )

        exits = (
            (k < d)
            &
            (k > overbought)
        )

        return entries, exits

    def generate_indicators(
        self,
        df,
        period,
        alpha_k,
        alpha_d
    ):

        low_min = (
            df["Low"]
            .rolling(window=period)
            .min()
        )

        high_max = (
            df["High"]
            .rolling(window=period)
            .max()
        )

        rsv = (
            (
                df["Close"]
                - low_min
            )
            /
            (
                high_max
                - low_min
                + 1e-9
            )
        ) * 100

        k = (
            rsv
            .ewm(
                alpha=alpha_k,
                adjust=False
            )
            .mean()
        )

        d = (
            k
            .ewm(
                alpha=alpha_d,
                adjust=False
            )
            .mean()
        )

        return {
            "K": k,
            "D": d
        }